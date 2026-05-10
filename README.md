# safetensors2GGUF

Konvertiert Safetensors-Modelle in das GGUF-Format für die Nutzung mit llama.cpp und ComfyUI-GGUF.

## Unterstützte Architekturen

| Architektur | Format |
|---|---|
| FLUX.1 | Diffusers |
| Stable Diffusion 3 | Diffusers |
| SDXL | Diffusers / Non-Diffusers |
| SD 1.x | Diffusers / Non-Diffusers |
| HiDream | Diffusers |
| HunyuanVideo | Diffusers |
| Wan | Diffusers |
| LTXV | Diffusers |
| Cosmos | Diffusers |
| AuraFlow | Diffusers |
| Lumina 2 | Diffusers |

## Installation

```bash
pip install -r requirements.txt
```

### Abhängigkeiten

```
gguf
torch
safetensors
tqdm
```

## Verwendung

```bash
python convert.py --src <pfad/zum/modell.safetensors> --dst <ausgabe.gguf>
```

### Optionen

| Option | Beschreibung |
|---|---|
| `--src` | Pfad zur Quelldatei (`.safetensors`, `.ckpt`, `.pt`, `.bin`) |
| `--dst` | Ausgabepfad für die GGUF-Datei (optional, wird automatisch generiert) |

### Ausgabeformat

Die erzeugte GGUF-Datei enthält:
- Tensoren in **F16** oder **BF16** (je nach Quelldtype)
- 1D-Tensoren und kleine Tensoren (≤ 1024 Elemente) immer in **F32**
- Metadaten für Architektur und Quantisierungsversion

### Nachbearbeitung von 5D-Tensoren (HunyuanVideo / Wan)

Einige Modelle enthalten 5D-Tensoren, die GGUF nicht direkt unterstützt.
Diese werden während der Konvertierung in eine separate Datei ausgelagert:

```bash
# Nach der Konvertierung:
python fix_5d_tensors.py --src <ausgabe.gguf> --dst <final.gguf>
```

## Bekannte Probleme

Siehe [Issues-Analyse](docs/issues_analysis.md) für häufige Fehler und deren Behebung.

## Lizenz

Apache-2.0
