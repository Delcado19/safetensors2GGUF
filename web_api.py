"""Loopback-only React workbench API; one conversion runs at a time.

Jobs are process-local. Restarting cancels cooperative workers and clears history.
The browser receives a session token from the same-origin config endpoint; every
filesystem/job request requires it, preventing unrelated sites from using this
local service as a filesystem API. No uploads or model downloads are performed.
"""
from __future__ import annotations

import argparse
import gc
import os
import secrets
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from conversion_service import _pipeline, _resolve_dst_st, _strip_model_suffix
from convert import ConversionCancelled, load_state_dict
from conversion_io import validate_output
from convert_safetensors import convert_to_safetensors
from models.architectures import detect_arch
from quantize import ALL_QUANT_CHOICES, LLAMA_QUANT_KEYS, estimate_output_size, find_exe
from safetensors_quant import SAFETENSORS_DTYPE_CHOICES, filename_suffix_for, safetensors_output_size_breakdown

MODEL_SUFFIXES = {'.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf'}
DIST = Path(__file__).parent / 'frontend' / 'dist'


class ConversionRequest(BaseModel):
    """Validated local conversion parameters; format keys come from the registry."""

    model_config = ConfigDict(extra='forbid')
    source: str = Field(min_length=1, max_length=4096)
    destination: str = Field(default='', max_length=4096)
    container: Literal['gguf', 'safetensors'] = 'gguf'
    format: str = 'Q4_K_M'
    precision_profile: Literal['auto', 'conservative', 'qwen_edit_2511', 'z_image_turbo'] = 'auto'
    overwrite: bool = False
    keep_intermediate: bool = False
    threads: int = Field(default=0, ge=0, le=1024)
    executable: str = Field(default='', max_length=4096)


class Job:
    """Locked snapshot of a cooperative conversion worker."""

    def __init__(self, parameters: ConversionRequest):
        self.id = uuid.uuid4().hex
        self.parameters = parameters
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.logs = deque(maxlen=400)
        self.status = 'queued'
        self.progress = 0.0
        self.phase = 'Preparing conversion'
        self.output = ''
        self.error = ''
        self.started = time.time()
        self.finished = None
        self.thread = None

    def snapshot(self):
        """Return a JSON-ready, internally consistent state."""
        with self.lock:
            return {'id': self.id, 'status': self.status, 'progress': self.progress,
                    'phase': self.phase, 'output': self.output, 'error': self.error,
                    'logs': list(self.logs), 'source': self.parameters.source,
                    'format': self.parameters.format, 'started': self.started,
                    'finished': self.finished}

    def emit(self, event):
        """Accept the existing pipeline's queue events without a second queue."""
        with self.lock:
            if event[0] == 'log':
                self.logs.append(str(event[1]))
            elif event[0] == 'progress_frac':
                self.progress = max(0.0, min(1.0, event[1]))
                self.phase = event[2]
            elif event[0] == 'progress':
                self.progress = event[1] / event[2] if event[2] else 0
                self.phase = f'Tensor {event[1]} of {event[2]}'

    def run(self):
        """Use existing atomic writers; cancellation remains cooperative."""
        params = self.parameters
        with self.lock:
            self.status = 'running'
        try:
            if self.cancel.is_set():
                raise ConversionCancelled('cancelled')
            if params.container == 'gguf':
                # The shared pipeline only needs queue.put; no polling thread.
                output = _pipeline(params.source, params.destination, params.format,
                                   params.executable, params.threads, params.keep_intermediate,
                                   params.overwrite, self, self.cancel)
            else:
                output, _ = convert_to_safetensors(
                    params.source, params.destination, target_key=params.format,
                    overwrite=params.overwrite, precision_profile=params.precision_profile,
                    cancel_event=self.cancel, log_tensor_every=25,
                    on_log=lambda message: self.emit(('log', message)),
                    on_progress=lambda index, total, key: self.emit(('progress', index, total, key)))
            with self.lock:
                # Publishing an output wins a late cancellation: report success
                # truthfully if the atomic writer already completed.
                self.output, self.status, self.progress = str(output), 'succeeded', 1.0
                self.phase = 'Conversion complete'
        except Exception as exc:
            with self.lock:
                cancelled = isinstance(exc, ConversionCancelled) or str(exc) == 'cancelled'
                self.status = 'cancelled' if cancelled else 'failed'
                self.error = '' if cancelled else str(exc)
                self.phase = 'Cancelled' if cancelled else 'Conversion failed'
        finally:
            gc.collect()
            with self.lock:
                self.finished = time.time()

    put = emit


def create_app() -> FastAPI:
    """Create an isolated API with bounded job history and same-origin access."""
    token = secrets.token_urlsafe(32)
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        """Signal cooperative cancellation when the API shuts down."""
        yield
        with jobs_lock:
            for job in jobs.values():
                job.cancel.set()

    app = FastAPI(title='safetensors workbench', lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', '[::1]'])

    def authorize(request: Request, x_workbench_token: str = Header(default='')):
        """Require a process-local token and reject foreign browser origins."""
        origin = request.headers.get('origin')
        # Vite development uses its own loopback origin; production is same origin.
        allowed = {str(request.base_url).rstrip('/'), 'http://127.0.0.1:5173', 'http://localhost:5173'}
        if origin and origin not in allowed:
            raise HTTPException(403, 'Untrusted origin')
        if not secrets.compare_digest(x_workbench_token, token):
            raise HTTPException(403, 'Missing or invalid workbench session')

    protected = [Depends(authorize)]

    @app.get('/api/config')
    def config(response: Response):
        """Return process-local configuration without caching the session token."""
        response.headers['Cache-Control'] = 'no-store'
        return {'token': token, 'formats': {'gguf': ALL_QUANT_CHOICES, 'safetensors': SAFETENSORS_DTYPE_CHOICES},
                'executable': str(find_exe() or ''), 'home': str(Path.home()),
                'platform': os.name, 'history_persistent': False}

    @app.get('/api/files', dependencies=protected)
    def files(path: str = '', directories_only: bool = False):
        """Browse local paths without uploading large checkpoints."""
        folder = Path(path).expanduser() if path else Path.home()
        try:
            folder = folder.resolve(strict=True)
            if not folder.is_dir():
                raise HTTPException(400, 'Choose a directory')
            entries = []
            for child in folder.iterdir():
                try:
                    directory = child.is_dir()
                    if directory or (not directories_only and child.suffix.lower() in MODEL_SUFFIXES):
                        entries.append({'name': child.name, 'path': str(child), 'directory': directory,
                                        'size': 0 if directory else child.stat().st_size})
                except OSError:
                    continue
            entries.sort(key=lambda entry: (not entry['directory'], entry['name'].lower()))
            return {'path': str(folder), 'parent': str(folder.parent), 'entries': entries[:1000],
                    'truncated': len(entries) > 1000}
        except OSError as exc:
            raise HTTPException(400, str(exc)) from exc

    def resolve(params):
        """Resolve output naming and enforce format/source/overwrite guards."""
        source = Path(params.source.strip()).expanduser()
        if not source.is_file() or source.suffix.lower() not in MODEL_SUFFIXES - {'.gguf'}:
            raise HTTPException(400, 'Choose an existing safetensors or checkpoint source')
        params.source = str(source.resolve())
        choices = ALL_QUANT_CHOICES if params.container == 'gguf' else SAFETENSORS_DTYPE_CHOICES
        if params.format not in {key for _, key in choices}:
            raise HTTPException(400, 'Unsupported quantization format')
        if params.container == 'gguf':
            from conversion_service import _resolve_dst
            params.destination = _resolve_dst(params.source, params.destination, params.format) or (
                str(_strip_model_suffix(params.source)) + f'-{params.format}.gguf')
        else:
            params.destination = _resolve_dst_st(params.source, params.destination, params.format) or (
                str(_strip_model_suffix(params.source)) + f'-{filename_suffix_for(params.format)}.safetensors')
        try:
            validate_output(params.source, params.destination, params.overwrite)
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return params

    @app.post('/api/inspect', dependencies=protected)
    def inspect(params: ConversionRequest):
        """Detect architecture and calculate source-aware payload estimates."""
        params = resolve(params)
        try:
            arch = detect_arch(load_state_dict(params.source))
            if params.container == 'gguf':
                size = estimate_output_size(params.source, params.format)
                breakdown = None
            else:
                breakdown = safetensors_output_size_breakdown(params.source, params.format, arch,
                                                             precision_profile=params.precision_profile)
                size = breakdown['total'] if breakdown else None
            return {'architecture': arch.arch, 'source_bytes': Path(params.source).stat().st_size,
                    'estimated_bytes': size, 'breakdown': breakdown, 'destination': params.destination}
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/jobs', status_code=202, dependencies=protected)
    def start(params: ConversionRequest):
        """Start one conversion, rejecting concurrent jobs to bound memory."""
        params = resolve(params)
        if params.container == 'gguf' and params.format in LLAMA_QUANT_KEYS:
            executable = Path(params.executable) if params.executable else find_exe()
            if not executable or not Path(executable).is_file():
                raise HTTPException(400, 'This format requires llama-quantize. Set its path in Advanced.')
            params.executable = str(executable)
        with jobs_lock:
            if any(job.snapshot()['status'] in {'queued', 'running', 'cancelling'} for job in jobs.values()):
                raise HTTPException(409, 'A conversion is already running. Wait or cancel it first.')
            # ponytail: one worker bounds model RAM; add scheduling only if needed.
            while len(jobs) >= 20:
                del jobs[next(iter(jobs))]
            job = Job(params)
            jobs[job.id] = job
            job.thread = threading.Thread(target=job.run, daemon=True)
            job.thread.start()
            return job.snapshot()

    @app.get('/api/jobs', dependencies=protected)
    def history():
        """Return recent jobs in newest-first order."""
        with jobs_lock:
            return [job.snapshot() for job in reversed(list(jobs.values()))]

    def get_job(job_id):
        """Look up a retained job or return HTTP 404."""
        with jobs_lock:
            if job_id not in jobs:
                raise HTTPException(404, 'Job not found')
            return jobs[job_id]

    @app.get('/api/jobs/{job_id}', dependencies=protected)
    def state(job_id: str):
        """Return an atomic worker snapshot."""
        return get_job(job_id).snapshot()

    @app.post('/api/jobs/{job_id}/cancel', dependencies=protected)
    def cancel(job_id: str):
        """Signal cooperative cancellation without deleting published outputs."""
        job = get_job(job_id)
        with job.lock:
            if job.status in {'queued', 'running', 'cancelling'}:
                job.cancel.set()
                job.status, job.phase = 'cancelling', 'Stopping safely…'
        return job.snapshot()

    @app.get('/')
    def index():
        """Serve the built workbench or explain the missing build."""
        if not (DIST / 'index.html').is_file():
            raise HTTPException(503, 'Build the frontend first: cd frontend && npm ci && npm run build')
        return FileResponse(DIST / 'index.html')

    if (DIST / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=DIST / 'assets'), name='assets')
    return app


app = create_app()

if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host='127.0.0.1', port=args.port)
