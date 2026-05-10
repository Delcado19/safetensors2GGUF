# Bekannte Konvertierungsfehler und Lösungen

Analyse der GitHub Issues von [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF/issues).

---

## 1. `inf`/`NaN`-Werte nach der Konvertierung

**Fehlermeldung:**
```
ggml_validate_row_data: found inf value at block 0
llama_model_quantize: failed to quantize: tensor 'norm_final.weight' has invalid data
```

**Ursache:** BF16-Werte im Quellmodell überschreiten nach der Konvertierung zu F16 den
darstellbaren Bereich (> 65504). `llama-quantize` verweigert die Quantisierung.

**Fix:**
```python
# In handle_tensors(), vor der dtype-Konvertierung:
data = torch.nan_to_num(data, nan=0.0, posinf=65504, neginf=-65504)
```

---

## 2. `size mismatch` beim Laden — BF16-1D-Tensoren

**Fehlermeldung:**
```
size mismatch for x_pad_token: copying a param with shape torch.Size([3840])
from checkpoint, the shape in current model is torch.Size([1, 3840])
```

**Ursache:** BF16-kodierte 1D-Tensoren werden beim Lesen aus GGUF als doppelt so viele
Bytes interpretiert (BF16 = 2 Bytes, aber als uint8 gelesen → scheinbar doppelt so lang).

**Betroffene Schlüssel:** `x_pad_token`, `cap_pad_token`, `.scale`-Norm-Gewichte.

**Fix:** Diese Schlüssel in `keys_hiprec` der Architekturklasse aufnehmen → werden als F32
gespeichert und sind immun gegen das Problem.

---

## 3. Falsche Architekturerkennung

**Fehlermeldung:**
```
AssertionError: Unknown model architecture!
# oder:
AssertionError: Model architecture not allowed for conversion!
```

**Ursachen:**
- Modell ist im **Reference-Format** statt Diffusers-Format (hat `keys_banned`-Schlüssel)
- Modell ist ein **Merge**, dessen Schlüssel leicht vom Standard abweichen
- Externe GGUF-Datei ohne `general.architecture`-Feld, Schlüsselabgleich schlägt fehl

**Diagnose:** `keys_detect` der entsprechenden Architekturklasse prüfen, ob die
Erkennungsschlüssel im State-Dict vorhanden sind.

---

## 4. `mat1 and mat2 shapes cannot be multiplied` — Matrizenmultiplikationsfehler

**Fehlermeldung:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (NxM and KxM)
```

**Ursache A — `shape_fix`-Reshape ohne Metadaten-Wiederherstellung:**
Für SD1/SDXL werden Tensoren auf `(n//256, 256)` umgeformt. Die Originalform wird als
`comfy.gguf.orig_shape.<key>`-Metadatenfeld gespeichert. Fehlt dieses Feld beim Laden,
bleibt der Tensor in der falschen Form → Matmul schlägt fehl.

**Ursache B — Falscher `shape_fix`-Einsatz:**
`shape_fix=True` darf nur für SD1/SDXL gesetzt werden. Bei anderen Architekturen
(Flux, SD3 etc.) führt es zu inkompatiblen Tensorformen.

**Fix:** `shape_fix` nur in `ModelSD1` und `ModelSDXL` aktivieren. `orig_shape`-Metadaten
immer beim Reshape mitschreiben.

---

## 5. `GGML_ASSERT(ne[i] > 0)` in llama-quantize

**Fehlermeldung:**
```
/ggml/src/ggml.c:22112: GGML_ASSERT(info->ne[i] > 0) failed
```

**Ursache:** Die von `convert.py` erzeugte GGUF-Datei enthält Tensoren mit einer
Dimension = 0, oder die Shape-Metadaten sind für llama.cpps Validator ungültig.
Häufig bei Merge-Modellen mit ungewöhnlichen Tensor-Shapes.

**Hinweis:** Das F16-GGUF wird erfolgreich erstellt — der Fehler tritt erst beim
zweiten Schritt (`llama-quantize`) auf.

---

## 6. `gather(): Expected dtype int64 for index`

**Fehlermeldung:**
```
gather(): Expected dtype int64 for index
```

**Ursache:** In der Dequantisierungslogik wird ein Index-Tensor mit `torch.int32`
erzeugt. PyTorch ≥ 2.6 erfordert `int64` für `torch.gather`.

**Fix:**
```python
# dequant.py ~Zeile 280:
# vorher:
qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int32))
# nachher:
qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64))
```

---

## 7. `only 0-dimensional arrays can be converted to Python scalars`

**Ursache:** Die `gguf`-Bibliothek gibt für skalare Metadatenfelder manchmal 1D-Arrays
der Größe 1 statt echter 0D-Skalare zurück. `.item()` schlägt dann fehl.

**Fix:** Beim Lesen von Metadatenfeldern prüfen ob das Array die Größe 1 hat,
bevor `.item()` aufgerufen wird.
