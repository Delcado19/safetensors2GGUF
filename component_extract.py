"""Analyze and extract embedded SDXL VAE/CLIP components from checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SDXL_COMPONENTS = ("vae", "clip_l", "clip_g")


@dataclass(frozen=True)
class ComponentAnalysis:
    """Comparison summary for one embedded checkpoint component."""

    name: str
    embedded_tensors: int
    output_tensors: int
    reference_path: str | None
    reference_tensors: int
    exact_matches: int
    mismatches: int
    missing_reference_keys: int
    extra_reference_keys: int
    max_abs_diff: float | None
    mean_abs_diff: float | None

    @property
    def has_reference(self) -> bool:
        return self.reference_path is not None

    @property
    def is_exact_standard(self) -> bool:
        return (
            self.has_reference
            and self.output_tensors > 0
            and self.exact_matches == self.output_tensors
            and self.mismatches == 0
            and self.missing_reference_keys == 0
        )

    @property
    def status(self) -> str:
        if self.output_tensors == 0:
            return "not found"
        if not self.has_reference:
            return "no local reference"
        if self.is_exact_standard:
            return "matches local standard"
        if self.missing_reference_keys == 0:
            return "differs from local standard"
        return "incomplete reference comparison"


@dataclass(frozen=True)
class ExtractedComponent:
    """File written for one extracted component."""

    name: str
    path: str
    tensors: int


def find_models_root(path: str | Path) -> Path | None:
    """Return the nearest ancestor named ``models`` for a ComfyUI model path."""

    current = Path(path).resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if parent.name.lower() == "models":
            return parent
    return None


def default_output_root(checkpoint_path: str | Path) -> Path:
    """Return the default destination root for extracted components."""

    return find_models_root(checkpoint_path) or Path(checkpoint_path).resolve().parent


def _component_output_dir(models_root: Path, component: str) -> Path:
    if component == "vae":
        return models_root / "vae"
    if component in {"clip_l", "clip_g"}:
        return models_root / "clip"
    raise ValueError(f"Unknown component: {component}")


def _component_reference_path(models_root: Path, component: str) -> Path:
    if component == "vae":
        return models_root / "vae" / "sdxlVAE.safetensors"
    if component == "clip_l":
        return models_root / "clip" / "clip_l.safetensors"
    if component == "clip_g":
        return models_root / "clip" / "clip_g.safetensors"
    raise ValueError(f"Unknown component: {component}")


def _direct_prefixed_keys(keys: list[str], prefix: str, skip: set[str] | None = None) -> dict[str, str]:
    skip = skip or set()
    return {
        key[len(prefix):]: key
        for key in keys
        if key.startswith(prefix) and key[len(prefix):] not in skip
    }


def _clip_g_mapping(keys: list[str]) -> dict[str, tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    """Map embedded OpenCLIP bigG keys to Comfy/HF CLIP-G keys."""

    key_set = set(keys)
    prefix = "conditioner.embedders.1.model."
    mapped: dict[str, tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = {}

    def add(src_suffix: str, dst_key: str, transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        src_key = prefix + src_suffix
        if src_key in key_set:
            mapped[dst_key] = (src_key, transform or (lambda tensor: tensor))

    add("token_embedding.weight", "text_model.embeddings.token_embedding.weight")
    add("positional_embedding", "text_model.embeddings.position_embedding.weight")
    add("ln_final.weight", "text_model.final_layer_norm.weight")
    add("ln_final.bias", "text_model.final_layer_norm.bias")
    add("text_projection", "text_projection.weight", lambda tensor: tensor.T)

    for index in range(64):
        base = f"transformer.resblocks.{index}"
        for packed, attr in (("attn.in_proj_weight", "weight"), ("attn.in_proj_bias", "bias")):
            src_key = prefix + f"{base}.{packed}"
            if src_key in key_set:
                for chunk_index, name in enumerate(("q_proj", "k_proj", "v_proj")):
                    mapped[f"text_model.encoder.layers.{index}.self_attn.{name}.{attr}"] = (
                        src_key,
                        lambda tensor, idx=chunk_index: tensor.chunk(3, dim=0)[idx],
                    )

        for src_suffix, dst_suffix in (
            ("attn.out_proj.weight", "self_attn.out_proj.weight"),
            ("attn.out_proj.bias", "self_attn.out_proj.bias"),
            ("ln_1.weight", "layer_norm1.weight"),
            ("ln_1.bias", "layer_norm1.bias"),
            ("ln_2.weight", "layer_norm2.weight"),
            ("ln_2.bias", "layer_norm2.bias"),
            ("mlp.c_fc.weight", "mlp.fc1.weight"),
            ("mlp.c_fc.bias", "mlp.fc1.bias"),
            ("mlp.c_proj.weight", "mlp.fc2.weight"),
            ("mlp.c_proj.bias", "mlp.fc2.bias"),
        ):
            add(f"{base}.{src_suffix}", f"text_model.encoder.layers.{index}.{dst_suffix}")

    return mapped


def _component_mapping(keys: list[str], component: str) -> dict[str, tuple[str, Callable[[torch.Tensor], torch.Tensor]]]:
    if component == "vae":
        return {
            dst: (src, lambda tensor: tensor)
            for dst, src in _direct_prefixed_keys(keys, "first_stage_model.").items()
        }
    if component == "clip_l":
        return {
            dst: (src, lambda tensor: tensor)
            for dst, src in _direct_prefixed_keys(
                keys,
                "conditioner.embedders.0.transformer.",
                skip={"text_model.embeddings.position_ids"},
            ).items()
        }
    if component == "clip_g":
        return _clip_g_mapping(keys)
    raise ValueError(f"Unknown component: {component}")


def analyze_components(
    checkpoint_path: str | Path,
    models_root: str | Path | None = None,
    components: tuple[str, ...] = SDXL_COMPONENTS,
) -> list[ComponentAnalysis]:
    """Analyze embedded SDXL components and compare them to local references."""

    checkpoint = Path(checkpoint_path)
    root = Path(models_root) if models_root else default_output_root(checkpoint)
    results: list[ComponentAnalysis] = []

    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        source_keys = list(source.keys())
        for component in components:
            mapping = _component_mapping(source_keys, component)
            reference = _component_reference_path(root, component)
            reference_path = str(reference) if reference.is_file() else None
            reference_tensors = 0
            exact = 0
            mismatches = 0
            missing_reference = 0
            extra_reference = 0
            max_abs_diff: float | None = None
            mean_abs_values: list[float] = []

            if reference_path:
                with safe_open(reference_path, framework="pt", device="cpu") as ref:
                    ref_keys = set(ref.keys())
                    reference_tensors = len(ref_keys)
                    for dst_key, (src_key, transform) in mapping.items():
                        if dst_key not in ref_keys:
                            missing_reference += 1
                            continue
                        src_tensor = transform(source.get_tensor(src_key)).contiguous()
                        ref_tensor = ref.get_tensor(dst_key)
                        if src_tensor.shape == ref_tensor.shape and torch.equal(src_tensor, ref_tensor):
                            exact += 1
                            continue
                        mismatches += 1
                        if src_tensor.shape == ref_tensor.shape:
                            diff = (src_tensor.float() - ref_tensor.float()).abs()
                            current_max = float(diff.max())
                            max_abs_diff = current_max if max_abs_diff is None else max(max_abs_diff, current_max)
                            mean_abs_values.append(float(diff.mean()))
                    extra_reference = len(ref_keys - set(mapping))

            results.append(
                ComponentAnalysis(
                    name=component,
                    embedded_tensors=len([
                        key for key in source_keys
                        if key.startswith(_source_prefix(component))
                    ]),
                    output_tensors=len(mapping),
                    reference_path=reference_path,
                    reference_tensors=reference_tensors,
                    exact_matches=exact,
                    mismatches=mismatches,
                    missing_reference_keys=missing_reference,
                    extra_reference_keys=extra_reference,
                    max_abs_diff=max_abs_diff,
                    mean_abs_diff=(
                        sum(mean_abs_values) / len(mean_abs_values)
                        if mean_abs_values else None
                    ),
                )
            )

    return results


def _source_prefix(component: str) -> str:
    if component == "vae":
        return "first_stage_model."
    if component == "clip_l":
        return "conditioner.embedders.0."
    if component == "clip_g":
        return "conditioner.embedders.1."
    raise ValueError(f"Unknown component: {component}")


def format_component_analysis(results: list[ComponentAnalysis]) -> str:
    """Format component analysis for logs or GUI text boxes."""

    if not results:
        return "No SDXL components found."

    lines: list[str] = []
    for result in results:
        lines.append(f"{result.name}: {result.status}")
        lines.append(
            f"  embedded: {result.embedded_tensors}, exportable: {result.output_tensors}"
        )
        if result.reference_path:
            lines.append(f"  reference: {result.reference_path}")
            lines.append(
                "  compare: "
                f"{result.exact_matches} exact, {result.mismatches} different, "
                f"{result.missing_reference_keys} missing, "
                f"{result.extra_reference_keys} reference-only"
            )
            if result.max_abs_diff is not None:
                lines.append(
                    f"  diff: max_abs={result.max_abs_diff:.6g}, "
                    f"mean_abs={result.mean_abs_diff:.6g}"
                )
        else:
            lines.append("  reference: not found")
        lines.append("")
    return "\n".join(lines).rstrip()


def extract_components(
    checkpoint_path: str | Path,
    models_root: str | Path | None = None,
    extract_vae: bool = False,
    extract_clip_l: bool = True,
    extract_clip_g: bool = True,
    overwrite: bool = False,
    on_log=None,
) -> list[ExtractedComponent]:
    """Extract selected embedded SDXL components into a ComfyUI models folder."""

    checkpoint = Path(checkpoint_path)
    root = Path(models_root) if models_root else default_output_root(checkpoint)
    selected = [
        ("vae", extract_vae),
        ("clip_l", extract_clip_l),
        ("clip_g", extract_clip_g),
    ]
    written: list[ExtractedComponent] = []

    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        keys = list(source.keys())
        stem = checkpoint
        while stem.suffix:
            stem = stem.with_suffix("")

        for component, enabled in selected:
            if not enabled:
                continue
            mapping = _component_mapping(keys, component)
            if not mapping:
                if on_log:
                    on_log(f"skip {component}: no embedded tensors found")
                continue

            output_dir = _component_output_dir(root, component)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{stem.name}-{component}.safetensors"
            if output_path.exists() and not overwrite:
                raise OSError(f"Output exists and overwrite is disabled: {output_path}")

            tensors = {
                dst_key: transform(source.get_tensor(src_key)).contiguous()
                for dst_key, (src_key, transform) in sorted(mapping.items())
            }
            save_file(tensors, str(output_path))
            written.append(
                ExtractedComponent(
                    name=component,
                    path=str(output_path),
                    tensors=len(tensors),
                )
            )
            if on_log:
                on_log(f"wrote {component}: {output_path} ({len(tensors)} tensors)")

    return written
