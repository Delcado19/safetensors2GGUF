# Architektur

## Konvertierungspipeline

```
Quelldatei (.safetensors / .ckpt / .pt)
    │
    ▼
load_state_dict()
    │  - Lädt Tensoren in den Speicher
    │  - Erkennt und entfernt State-Dict-Präfixe (z.B. "model.diffusion_model.")
    │
    ▼
detect_arch()
    │  - Gleicht Schlüssel im State-Dict gegen keys_detect jeder Architekturklasse ab
    │  - Wirft AssertionError wenn keine Architektur erkannt wird
    │
    ▼
handle_tensors()
    │  - Iteriert über alle Tensoren
    │  - Filtert keys_ignore
    │  - Konvertiert Dtype: BF16 → F32, float8 → F16
    │  - Entscheidet Quantisierungstyp:
    │      1D oder ≤ 1024 Elemente oder keys_hiprec → F32
    │      BF16-Quelle → BF16
    │      sonst → F16
    │  - Reshape für SD1/SDXL (shape_fix): (H,W) → (n//256, 256)
    │      speichert orig_shape als Metadatenfeld
    │  - Verarbeitet 5D-Tensoren: auslagern statt schreiben
    │
    ▼
GGUFWriter
    │  - Schreibt Header (arch, file_type, quantization_version)
    │  - Schreibt KV-Metadaten
    │  - Schreibt Tensordaten
    │
    ▼
Ausgabedatei (.gguf)
```

## Architekturklassen

Jede unterstützte Modellarchitektur erbt von `ModelTemplate`:

```python
class ModelTemplate:
    arch = "invalid"       # GGUF-Architekturstring
    shape_fix = False      # Nur True für SD1/SDXL
    keys_detect = []       # Schlüssellisten zur Erkennung
    keys_banned = []       # Ungültige Varianten (z.B. Reference-Format)
    keys_hiprec = []       # Tensoren die F32 brauchen
    keys_ignore = []       # Tensoren die übersprungen werden
```

### Erkennungslogik

`keys_detect` ist eine Liste von Tupeln. Ein Modell wird erkannt wenn **alle** Schlüssel
eines Tupels im State-Dict vorhanden sind:

```python
# Flux wird erkannt wenn EINES dieser Tupel vollständig passt:
keys_detect = [
    ("transformer_blocks.0.attn.norm_added_k.weight",),          # Diffusers
    ("double_blocks.0.img_attn.proj.weight",),                    # Non-Diffusers
]
# Gleichzeitig: Reference-Format ist verboten:
keys_banned = ["transformer_blocks.0.attn.norm_added_k.weight"]
```

## 5D-Tensor-Handling

GGUF unterstützt maximal 4-dimensionale Tensoren. Modelle wie HunyuanVideo und Wan
enthalten vereinzelt 5D-Tensoren (z.B. RoPE-Frequenzen).

**Zweistufiger Prozess:**

1. `convert.py`: 5D-Tensor wird **nicht** in die GGUF-Datei geschrieben, sondern in
   `fix_5d_tensors_<arch>.safetensors` ausgelagert.

2. `fix_5d_tensors.py`: Liest die fertig quantisierte GGUF-Datei und fügt den
   ausgelagerten Tensor als F32 ein.

## Quantisierungsentscheidungsbaum

```
Tensor
 ├─ in keys_ignore?          → überspringen
 ├─ ndim > 4?                → auslagern (5D-Fix)
 ├─ ndim == 1?               → F32
 ├─ n_params ≤ 1024?         → F32
 ├─ key in keys_hiprec?      → F32
 ├─ shape_fix anwendbar?     → reshape + orig_shape-Metadatum schreiben
 └─ dtype der Quelle?
      BF16 → BF16
      sonst → F16
```
