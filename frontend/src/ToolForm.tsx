import { useState } from "react";
import { ArrowRight, FolderOpen, ShieldCheck } from "lucide-react";
import { FormatSelect } from "./FormatSelect";

type Operation =
  "components" | "diffusion" | "pad_tokens" | "restore_5d";
const operations: [Operation, string, string][] = [
  [
    "components",
    "Extract components",
    "Check embedded SDXL VAE, CLIP-L and CLIP-G, reuse identical local references or export separate files.",
  ],
  [
    "diffusion",
    "Extract diffusion model",
    "Keep only diffusion weights, convert them and save to diffusion_models/.",
  ],
  [
    "pad_tokens",
    "Repair pad tokens",
    "Correct older Lumina2 GGUF pad-token shapes from [D] to [1, D].",
  ],
  [
    "restore_5d",
    "Restore 5D tensors",
    "Reinsert original 5D tensors from the conversion sidecar into a quantized GGUF.",
  ],
];

export function ToolForm({
  source,
  destination,
  setSource,
  setDestination,
  browse,
  formats,
  disabled,
  submit,
}: {
  source: string;
  destination: string;
  setSource: (value: string) => void;
  setDestination: (value: string) => void;
  browse: (target: "source" | "destination") => void;
  formats?: Record<"gguf" | "safetensors", [string, string][]>;
  disabled: boolean;
  submit: (parameters: unknown) => void;
}) {
  const [operation, setOperation] = useState<Operation>("components");
  const [components, setComponents] = useState(["clip_l", "clip_g"]);
  const [sidecar, setSidecar] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [reuseIdentical, setReuseIdentical] = useState(true);
  const [container, setContainer] = useState<"gguf" | "safetensors">(
    "safetensors",
  );
  const [format, setFormat] = useState("F16");
  const repair = operation === "pad_tokens" || operation === "restore_5d";
  const label = operations.find(([key]) => key === operation)!;
  return (
    <section className="conversion-panel" aria-labelledby="tool-title">
      <div className="panel-heading">
        <h2 id="tool-title">Extract & repair</h2>
        <span className="tag">MODEL TOOLS</span>
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          submit({
            operation,
            source,
            destination,
            sidecar: operation === "restore_5d" ? sidecar : "",
            components,
            overwrite,
            reuse_identical: reuseIdentical,
            container,
            format,
          });
        }}
      >
        <fieldset disabled={disabled}>
          <legend className="section-label">
            <span>01</span>Choose your task
          </legend>
          <label className="field-label" htmlFor="tool-operation">
            Operation
          </label>
          <select
            id="tool-operation"
            value={operation}
            onChange={(event) => {
              setOperation(event.target.value as Operation);
              setDestination("");
            }}
          >
            {operations.map(([key, name]) => (
              <option key={key} value={key}>
                {name}
              </option>
            ))}
          </select>
          <p className="encoder-note">{label[2]}</p>
          <label className="field-label" htmlFor="tool-source">
            {repair ? "Source GGUF" : "Source checkpoint"}
          </label>
          <div className="path-field">
            <input
              id="tool-source"
              required
              value={source}
              onChange={(event) => setSource(event.target.value)}
              placeholder={
                repair
                  ? "Path to model.gguf"
                  : "Path to SDXL checkpoint.safetensors"
              }
            />
            <button
              type="button"
              className="secondary"
              onClick={() => browse("source")}
              aria-label="Browse tool source"
            >
              <FolderOpen size={17} />
            </button>
          </div>
          <label className="field-label" htmlFor="tool-destination">
            {repair ? "Output GGUF or folder" : "ComfyUI models root"}
          </label>
          <div className="path-field">
            <input
              id="tool-destination"
              value={destination}
              onChange={(event) => setDestination(event.target.value)}
              placeholder={
                repair
                  ? "Automatic: source-fixed.gguf"
                  : "Automatic: nearest models folder or checkpoint folder"
              }
            />
            <button
              type="button"
              className="secondary"
              onClick={() => browse("destination")}
              aria-label="Browse tool output"
            >
              <FolderOpen size={17} />
            </button>
          </div>
          <p className="preview-note">
            {repair
              ? "Writes a separate file. The source is preserved."
              : "VAE goes to vae/, CLIP to clip/, diffusion to diffusion_models/."}
          </p>
          {operation === "components" && (
            <>
            <div className="tool-components">
              {[
                ["vae", "VAE"],
                ["clip_l", "CLIP-L"],
                ["clip_g", "CLIP-G"],
              ].map(([key, name]) => (
                <label className="checkbox" key={key}>
                  <input
                    type="checkbox"
                    checked={components.includes(key)}
                    onChange={(event) =>
                      setComponents(
                        event.target.checked
                          ? [...components, key]
                          : components.filter((item) => item !== key),
                      )
                    }
                  />
                  {name}
                </label>
              ))}
            </div>
            <label className="checkbox tool-overwrite">
              <input type="checkbox" checked={reuseIdentical}
                onChange={(event) => setReuseIdentical(event.target.checked)} />
              Reuse identical local components
            </label>
            <p className="preview-note">
              Checks sdxlVAE.safetensors in vae/ and clip_l.safetensors / clip_g.safetensors in clip/.
              Known SDXL stock components are recognized even without local references.
              Reuse requires an identical local file; other components are exported.
            </p>
            </>
          )}
          {operation === "restore_5d" && (
            <>
              <label className="field-label" htmlFor="tool-sidecar">
                5D sidecar (optional)
              </label>
              <input
                id="tool-sidecar"
                value={sidecar}
                onChange={(event) => setSidecar(event.target.value)}
                placeholder="Automatic: GGUF metadata, then legacy filenames"
              />
            </>
          )}
          {operation === "diffusion" && (
            <>
              <label className="field-label" htmlFor="tool-container">
                Output container
              </label>
              <select
                id="tool-container"
                value={container}
                onChange={(event) => {
                  setContainer(event.target.value as typeof container);
                  setFormat("F16");
                }}
              >
                <option value="safetensors">Safetensors</option>
                <option value="gguf">GGUF</option>
              </select>
              <label className="field-label" htmlFor="tool-format">
                Quantization
              </label>
              <FormatSelect
                id="tool-format"
                value={format}
                onChange={setFormat}
                container={container}
                choices={formats?.[container] || []}
              />
            </>
          )}
            <label className="checkbox tool-overwrite">
              <input
                type="checkbox"
                checked={overwrite}
                onChange={(event) => setOverwrite(event.target.checked)}
              />
              Replace existing output files
            </label>
        </fieldset>
        <div className="form-actions">
          <span>
            <ShieldCheck size={15} />
            Original stays untouched
          </span>
          <button
            className="primary"
            disabled={
              disabled ||
              !source.trim() ||
              (operation === "components" && !components.length)
            }
            type="submit"
          >
            {disabled ? "Please wait…" : label[1]}
            <ArrowRight size={17} />
          </button>
        </div>
      </form>
    </section>
  );
}
