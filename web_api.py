"""Loopback-only React workbench API; one model job runs at a time.

Jobs are process-local. Restarting cancels cooperative workers and clears history.
The browser receives a session token from the same-origin config endpoint; every
filesystem/job request requires it, preventing unrelated sites from using this
local service as a filesystem API. Hugging Face downloads are explicitly started
by the user; model files are never uploaded.
"""
from __future__ import annotations

import argparse
import gc
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import asdict
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
from component_extract import analyze_components, default_output_root, extract_components
from fix_pad_tokens import fix_pad_tokens
from fix_5d_tensors import fix_5d_tensors
from hf_download import download_repo_as_single_safetensors, inspect_repo, _hub_token, _validate_relative_path
from huggingface_hub import get_token
from huggingface_hub.utils import validate_repo_id
from models.architectures import detect_arch
from model_support import SUPPORT_BAD, text_encoder_support_level, text_encoder_support_reason
from model_support import build_workbench_support_tables
from quantize import ALL_QUANT_CHOICES, LLAMA_QUANT_KEYS, estimate_output_size, find_exe
from safetensors_quant import SAFETENSORS_DTYPE_CHOICES, filename_suffix_for, safetensors_output_size_breakdown
from text_encoder_convert import (
    TEXT_ENCODER_FORMAT_CHOICES, TEXT_ENCODER_SAFETENSORS_FORMATS,
    _TEXT_ENCODER_MODEL_ARCH, _TEXT_ENCODER_SAFETENSORS_TARGET_KEY,
    _VENDORED_REPOS, convert_text_encoder_any, detect_text_encoder_family,
)

MODEL_SUFFIXES = {'.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf'}
DIST = Path(__file__).parent / 'frontend' / 'dist'


class ConversionRequest(BaseModel):
    """Validated local conversion parameters; format keys come from the registry."""

    model_config = ConfigDict(extra='forbid')
    model_kind: Literal['diffusion', 'text_encoder'] = 'diffusion'
    base_repo_id: str = Field(default='', max_length=256)
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
    """Locked snapshot of a cooperative conversion, tool or download worker."""

    def __init__(self, parameters: ConversionRequest | ToolRequest | DownloadRequest):
        self.id = uuid.uuid4().hex
        self.parameters = parameters
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.logs = deque(maxlen=400)
        self.status = 'queued'
        self.progress = 0.0
        self.phase = 'Preparing job'
        self.output = ''
        self.error = ''
        self.started = time.time()
        self.finished = None
        self.thread = None
        self.outputs = []
        self.result = None

    def snapshot(self):
        """Return a JSON-ready, internally consistent state."""
        with self.lock:
            return {'id': self.id, 'status': self.status, 'progress': self.progress,
                    'phase': self.phase, 'output': self.output, 'error': self.error,
                    'logs': list(self.logs), 'source': self.parameters.source,
                    'format': self.parameters.format, 'started': self.started,
                    'model_kind': self.parameters.model_kind,
                    'operation': getattr(self.parameters, 'operation', 'convert'),
                    'outputs': list(self.outputs), 'result': self.result,
                    'indeterminate': (self.parameters.model_kind == 'text_encoder' or
                                      getattr(self.parameters, 'operation', '') in {'components', 'analyze', 'download'}) and self.status in {'queued', 'running', 'cancelling'},
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
            operation = getattr(params, 'operation', 'convert')
            if operation not in {'analyze', 'components', 'download'}:
                # Folder selections may name a new ComfyUI output directory.
                Path(params.destination).parent.mkdir(parents=True, exist_ok=True)
            callbacks = {'on_log': lambda message: self.emit(('log', message)),
                         'cancel_event': self.cancel}
            if operation == 'download':
                def transfer_phase(index, total, message):
                    # Counts describe shards, not bytes: never show a fake percentage.
                    with self.lock:
                        self.phase = f'{message} ({index + 1}/{total})'
                revision = params.revision if re.fullmatch(r'[0-9a-fA-F]{40}', params.revision) else inspect_repo(params.source, params.revision)['revision']
                output = download_repo_as_single_safetensors(params.source, params.destination,
                    subfolder=params.subfolder, revision=revision,
                    overwrite=params.overwrite, on_progress=transfer_phase, **callbacks)
            elif operation == 'analyze':
                results = analyze_components(params.source, params.destination, cancel_event=self.cancel)
                with self.lock:
                    self.result = [asdict(item) | {'status': item.status} for item in results]
                output = ''
            elif operation == 'components':
                def published(path):
                    # Keep completed component paths visible even after cancellation.
                    with self.lock:
                        self.outputs.append(path)
                written = extract_components(params.source, params.destination,
                    extract_vae='vae' in params.components,
                    extract_clip_l='clip_l' in params.components,
                    extract_clip_g='clip_g' in params.components,
                    overwrite=params.overwrite, on_output=published, **callbacks)
                if not written:
                    raise ValueError('No selected embedded SDXL components found')
                output = '\n'.join(item.path for item in written)
            elif operation in {'pad_tokens', 'restore_5d'}:
                callbacks['on_progress'] = lambda index, total, key: self.emit(('progress', index, total, key))
                if operation == 'pad_tokens':
                    fix_pad_tokens(params.source, params.destination, overwrite=params.overwrite, **callbacks)
                else:
                    fix_5d_tensors(params.source, params.destination, fix_path=params.sidecar or None,
                                   overwrite=params.overwrite, **callbacks)
                output = params.destination
            elif params.model_kind == 'text_encoder':
                output = convert_text_encoder_any(
                    params.source, params.base_repo_id, params.destination, params.format,
                    on_log=lambda message: self.emit(('log', message)),
                    cancel_event=self.cancel, overwrite=params.overwrite)
            elif params.container == 'gguf':
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
                self.phase = 'Analysis complete' if operation == 'analyze' else 'Job complete'
        except Exception as exc:
            with self.lock:
                cancelled = isinstance(exc, ConversionCancelled) or str(exc) == 'cancelled'
                self.status = 'cancelled' if cancelled else 'failed'
                self.error = '' if cancelled else (download_error(exc) if getattr(params, 'operation', '') == 'download' else str(exc))
                self.phase = 'Cancelled' if cancelled else 'Job failed'
        finally:
            gc.collect()
            with self.lock:
                self.finished = time.time()

    put = emit


class ToolRequest(BaseModel):
    """Local extraction/repair inputs; destination means models root for extraction."""

    model_config = ConfigDict(extra='forbid')
    operation: Literal['analyze', 'components', 'diffusion', 'pad_tokens', 'restore_5d']
    source: str = Field(min_length=1, max_length=4096)
    destination: str = Field(default='', max_length=4096)
    sidecar: str = Field(default='', max_length=4096)
    components: list[Literal['vae', 'clip_l', 'clip_g']] = Field(default_factory=lambda: ['clip_l', 'clip_g'], max_length=3)
    overwrite: bool = False
    container: Literal['gguf', 'safetensors'] = 'safetensors'
    format: str = 'F16'
    model_kind: Literal['diffusion'] = 'diffusion'


class DownloadRequest(BaseModel):
    """Hub model repo and local output folder; no browser-supplied credentials."""

    model_config = ConfigDict(extra='forbid')
    source: str = Field(min_length=1, max_length=256)
    destination: str = Field(default='', max_length=4096)
    revision: str = Field(default='main', min_length=1, max_length=256)
    subfolder: str | None = Field(default=None, max_length=1024)
    overwrite: bool = False
    operation: Literal['download'] = 'download'
    model_kind: Literal['diffusion'] = 'diffusion'
    format: Literal['safetensors'] = 'safetensors'


def download_error(exc: Exception) -> str:
    """Expose actionable download errors without HTTP headers, tokens or signed URLs."""
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status in {401, 403}:
        return 'Hugging Face access denied. Configure a server-side token and accept any model access conditions.'
    if status == 404:
        return 'Hugging Face repository, revision or file not found, or not accessible to your account.'
    if status == 429:
        return 'Hugging Face rate limit reached. Retry later.'
    if (isinstance(exc, ValueError) or type(exc) in {RuntimeError, OSError, FileExistsError, PermissionError}) and status is None:
        message = str(exc)
        token = _hub_token()
        return message.replace(token, '[redacted]') if token else message
    return 'Could not contact Hugging Face. Check the repository, access and connection, then retry.'


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

    @app.get('/api/support', dependencies=protected)
    def support():
        """Expose shared project classifications and reasons, without new compatibility claims."""
        return build_workbench_support_tables()

    @app.get('/api/config')
    def config(response: Response):
        """Return process-local configuration without caching the session token."""
        response.headers['Cache-Control'] = 'no-store'
        return {'token': token, 'formats': {'gguf': ALL_QUANT_CHOICES, 'safetensors': SAFETENSORS_DTYPE_CHOICES},
                'text_encoder_formats': {
                    container: [(label, key) for label, key in TEXT_ENCODER_FORMAT_CHOICES
                                if (key in TEXT_ENCODER_SAFETENSORS_FORMATS) == (container == 'safetensors')]
                    for container in ('gguf', 'safetensors')},
                'executable': str(find_exe() or ''), 'home': str(Path.home()),
                'platform': os.name, 'history_persistent': False,
                'hf_authenticated': bool(_hub_token() or get_token())}

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
        choices = TEXT_ENCODER_FORMAT_CHOICES if params.model_kind == 'text_encoder' else (
            ALL_QUANT_CHOICES if params.container == 'gguf' else SAFETENSORS_DTYPE_CHOICES)
        if params.format not in {key for _, key in choices}:
            raise HTTPException(400, 'Unsupported quantization format')
        if params.model_kind == 'text_encoder':
            if (params.format in TEXT_ENCODER_SAFETENSORS_FORMATS) != (params.container == 'safetensors'):
                raise HTTPException(400, 'Text-encoder format does not match the output container')
            if params.precision_profile != 'auto' or params.executable or params.threads or params.keep_intermediate:
                raise HTTPException(400, 'Diffusion advanced settings do not apply to text encoders')
            from conversion_service import _resolve_dst_te
            params.destination = _resolve_dst_te(params.source, params.destination, params.format) or (
                str(_strip_model_suffix(params.source)) + f'-{filename_suffix_for(params.format)}.{params.container}')
        elif params.container == 'gguf':
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

    def text_encoder_details(params):
        """Inspect original prefixes and reject known ComfyUI-incompatible outputs offline."""
        detected = detect_text_encoder_family(load_state_dict(params.source, strip_prefixes=False))
        manual = _VENDORED_REPOS.get(params.base_repo_id.strip()) if params.container == 'gguf' else None
        if manual and detected and manual != detected:
            raise ValueError(f'Base model override ({manual}) does not match detected weights ({detected}).')
        family = manual or detected
        key = ('GGUF' if params.container == 'gguf' else
               _TEXT_ENCODER_SAFETENSORS_TARGET_KEY.get(params.format, params.format))
        support = text_encoder_support_level(family, key) if family else 'unknown'
        reason = text_encoder_support_reason(family, key) if family else None
        if support == SUPPORT_BAD:
            raise ValueError((reason or f'{family} is not supported in {params.format} by ComfyUI.') +
                             ' Choose F16 safetensors or a supported format.')
        if params.container == 'gguf' and not family and not params.base_repo_id.strip():
            raise ValueError('Unknown text-encoder family. Enter the original base model repo ID or use safetensors.')
        return detected or family or 'Unknown family', support, reason

    @app.post('/api/inspect', dependencies=protected)
    def inspect(params: ConversionRequest):
        """Detect architecture and calculate source-aware payload estimates."""
        params = resolve(params)
        try:
            if params.model_kind == 'text_encoder':
                family, support, reason = text_encoder_details(params)
                # llama.cpp has a different writer: do not apply diffusion GGUF estimates.
                breakdown = (safetensors_output_size_breakdown(
                    params.source, _TEXT_ENCODER_SAFETENSORS_TARGET_KEY.get(params.format, params.format),
                    _TEXT_ENCODER_MODEL_ARCH, strip_prefixes=False)
                    if params.container == 'safetensors' else None)
                return {'architecture': family, 'source_bytes': Path(params.source).stat().st_size,
                        'estimated_bytes': breakdown['total'] if breakdown else None,
                        'destination': params.destination, 'support': support, 'support_reason': reason}
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
        if params.model_kind == 'text_encoder':
            try:
                text_encoder_details(params)
            except Exception as exc:
                # Invalid checkpoint headers must surface as input errors,
                # including format-specific exceptions from safetensors/torch.
                raise HTTPException(400, str(exc)) from exc
        if params.model_kind == 'diffusion' and params.container == 'gguf' and params.format in LLAMA_QUANT_KEYS:
            executable = Path(params.executable) if params.executable else find_exe()
            if not executable or not Path(executable).is_file():
                raise HTTPException(400, 'This format requires llama-quantize. Set its path in Advanced.')
            params.executable = str(executable)
        return enqueue(params)

    def enqueue(params):
        """Share the single-worker limit across conversion, extraction and repair."""
        with jobs_lock:
            if any(job.snapshot()['status'] in {'queued', 'running', 'cancelling'} for job in jobs.values()):
                raise HTTPException(409, 'A job is already running. Wait or cancel it first.')
            # ponytail: one worker bounds model RAM; add scheduling only if needed.
            while len(jobs) >= 20:
                del jobs[next(iter(jobs))]
            job = Job(params)
            jobs[job.id] = job
            job.thread = threading.Thread(target=job.run, daemon=True)
            job.thread.start()
            return job.snapshot()

    def resolve_download(params):
        """Reject unsafe repo paths and resolve an optional local output directory."""
        try:
            params.source = params.source.strip()
            validate_repo_id(params.source)
            params.revision = params.revision.strip()
            if not params.revision:
                raise ValueError('Enter a revision or use main')
            if params.subfolder is not None:
                params.subfolder = params.subfolder.strip()
                _validate_relative_path(params.subfolder)
            if params.destination.strip():
                folder = Path(params.destination.strip()).expanduser()
                if folder.exists() and not folder.is_dir():
                    raise ValueError('Choose an output directory')
                params.destination = str(folder.resolve())
        except (OSError, ValueError) as exc:
            raise HTTPException(400, download_error(exc)) from exc
        return params

    @app.post('/api/hf/inspect', dependencies=protected)
    def inspect_hf(params: DownloadRequest):
        """List checkpoint folders and pin the revision without downloading tensors."""
        params = resolve_download(params)
        try:
            return inspect_repo(params.source, params.revision)
        except Exception as exc:
            raise HTTPException(400, download_error(exc)) from exc

    @app.post('/api/hf/download', status_code=202, dependencies=protected)
    def download_hf(params: DownloadRequest):
        """Run the existing streaming downloader under the shared job/cancel limit."""
        params = resolve_download(params)
        if not params.destination:
            raise HTTPException(400, 'Choose an output directory')
        return enqueue(params)

    @app.post('/api/tools', status_code=202, dependencies=protected)
    def tools(params: ToolRequest):
        """Validate tool-specific paths and run existing backends in the shared worker."""
        source = Path(params.source.strip()).expanduser()
        repair = params.operation in {'pad_tokens', 'restore_5d'}
        if not source.is_file() or source.suffix.lower() != ('.gguf' if repair else '.safetensors'):
            raise HTTPException(400, 'Choose an existing GGUF' if repair else 'Choose an existing safetensors checkpoint')
        params.source = str(source.resolve())
        if params.sidecar and params.operation != 'restore_5d':
            raise HTTPException(400, 'A sidecar only applies to 5D restoration')
        try:
            if repair:
                destination = Path(params.destination.strip()).expanduser() if params.destination.strip() else source.with_name(source.stem + '-fixed.gguf')
                if destination.is_dir() or params.destination.endswith(('/', '\\')):
                    destination /= source.stem + '-fixed.gguf'
                if destination.suffix.lower() != '.gguf':
                    raise ValueError('Repair output must be a GGUF file')
                params.destination = str(destination.resolve())
                validate_output(source, destination, params.overwrite)
                if params.sidecar.strip():
                    sidecar = Path(params.sidecar.strip()).expanduser()
                    if not sidecar.is_file() or sidecar.suffix.lower() != '.safetensors':
                        raise ValueError('Choose an existing safetensors sidecar')
                    params.sidecar = str(sidecar.resolve())
                    validate_output(sidecar, destination, params.overwrite)
                else:
                    params.sidecar = ''
            else:
                root = Path(params.destination.strip()).expanduser() if params.destination.strip() else default_output_root(source)
                if root.exists() and not root.is_dir():
                    raise ValueError('Choose a models root directory')
                params.destination = str(root.resolve())
                if params.operation == 'components' and not params.components:
                    raise ValueError('Select at least one component')
                if params.operation == 'diffusion':
                    # Conversion already filters checkpoint keys to the diffusion model.
                    return start(ConversionRequest(source=params.source,
                        destination=str(root / 'diffusion_models') + os.sep,
                        container=params.container, format=params.format, overwrite=params.overwrite))
        except (OSError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return enqueue(params)

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
