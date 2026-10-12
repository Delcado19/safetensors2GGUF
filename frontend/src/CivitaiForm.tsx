import { useEffect, useRef, useState } from "react";
import { FolderOpen, Search } from "lucide-react";

type Inspection = {
  version: number;
  selected_file: number | null;
  files: { id: number; name: string; bytes: number | null; sha256: string }[];
};

export function CivitaiForm({ api, destination, setDestination, browse, disabled, submit, bytes, authenticated, configureCredentials }: {
  api: <T>(path: string, body?: unknown, signal?: AbortSignal) => Promise<T>;
  destination: string;
  setDestination: (value: string) => void;
  browse: () => void;
  disabled: boolean;
  submit: (parameters: unknown) => void;
  bytes: (value: number) => string;
  authenticated: boolean | undefined;
  configureCredentials: () => void;
}) {
  const [source, setSource] = useState("");
  const [inspection, setInspection] = useState<Inspection>();
  const [fileId, setFileId] = useState(0);
  const [overwrite, setOverwrite] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [error, setError] = useState("");
  const version = useRef(0);
  const request = useRef<AbortController | undefined>(undefined);
  useEffect(() => {
    version.current++;
    request.current?.abort();
    setInspection(undefined);
    setFileId(0);
    setInspecting(false);
    setError("");
    return () => { version.current++; request.current?.abort(); };
  }, [source]);
  async function inspect() {
    const current = ++version.current;
    request.current?.abort();
    request.current = new AbortController();
    setInspecting(true);
    setError("");
    try {
      const result = await api<Inspection>("civitai/inspect", { source }, request.current.signal);
      if (current === version.current) {
        setInspection(result);
        setFileId(result.selected_file || (result.files.length === 1 ? result.files[0].id : 0));
      }
    } catch (reason) {
      if (current === version.current && (reason as Error).name !== "AbortError")
        setError((reason as Error).message);
    } finally {
      if (current === version.current) setInspecting(false);
    }
  }
  const selected = inspection?.files.find((file) => file.id === fileId);
  return <section className="conversion-panel" aria-labelledby="civitai-title">
    <div className="panel-heading"><h2 id="civitai-title">Download from Civitai</h2><span className="tag">CIVITAI</span></div>
    <form onSubmit={(event) => {
      event.preventDefault();
      // Credential setup is required for download, while metadata inspection stays anonymous.
      if (authenticated === true && inspection && selected) submit({
        source: String(inspection.version), file_id: selected.id, destination, overwrite,
        format: selected.name.toLowerCase().endsWith(".gguf") ? "gguf" : "safetensors",
      });
    }}>
      <fieldset disabled={disabled}>
        <legend className="section-label"><span>01</span>Find your checkpoint</legend>
        <label className="field-label" htmlFor="civitai-source">Version ID or model/download link</label>
        <input id="civitai-source" value={source} onChange={(event) => setSource(event.target.value)}
          placeholder="https://civitai.red/api/download/models/…?fileId=…" spellCheck={false} autoCapitalize="none" />
        <button type="button" className="secondary inspect-button" disabled={!source.trim() || inspecting} onClick={inspect}>
          <Search size={16} />{inspecting ? "Reading model…" : "Inspect model"}
        </button>
        <p className="preview-note">Use a specific version. Reads metadata only; weights download when you start.</p>
        {error && <p className="picker-error" role="alert">{error}</p>}
        {inspection && <>
          <label className="field-label" htmlFor="civitai-file">Model file</label>
          <select id="civitai-file" value={fileId} onChange={(event) => setFileId(Number(event.target.value))}>
            <option value={0} disabled>Choose one model file</option>
            {inspection.files.map((file) => <option key={file.id} value={file.id}>{file.name}</option>)}
          </select>
        </>}
      </fieldset>
      <fieldset disabled={disabled}>
        <legend className="section-label"><span>02</span>Bring it home</legend>
        <label className="field-label" htmlFor="civitai-destination">Download folder</label>
        <div className="path-field">
          <input id="civitai-destination" required value={destination} onChange={(event) => setDestination(event.target.value)}
            placeholder="Choose a local folder" spellCheck={false} />
          <button className="secondary" type="button" onClick={browse} aria-label="Browse download folder"><FolderOpen size={17} /></button>
        </div>
        {selected && <div className="encoder-note hf-plan" role="status">
          <strong>{selected.name}</strong>
          <span>{selected.bytes == null ? "Download size unavailable" : bytes(selected.bytes)}</span>
          <small>Version {inspection!.version} · SHA256 verified before publication</small>
        </div>}
        <label className="checkbox tool-overwrite"><input type="checkbox" checked={overwrite}
          onChange={(event) => setOverwrite(event.target.checked)} />Replace an existing model file</label>
        <p className="preview-note">A Civitai API key is required. Cancel or connection failure keeps the partial file.
          Start the same file in the same folder to resume, including after a server restart.
          Temporary failures retry automatically with a fresh link and alternate domain.</p>
      </fieldset>
      <div className="credential-download-status" role="status" aria-live="polite">
        <p className="preview-note">{authenticated === true
          ? "API key configured. Model access is checked when downloading."
          : authenticated === false ? "Configure a Civitai API key before downloading."
          : "Credential status is being checked or needs recovery. Open API key setup to continue."}</p>
        {authenticated !== true && <button className="secondary" type="button" onClick={configureCredentials}>
          Configure API key
        </button>}
      </div>
      <div className="form-actions"><span>Publish only after SHA256 verification</span>
        <button className="primary" type="submit" disabled={disabled || authenticated !== true || inspecting || !selected || !destination.trim()}>
          {disabled ? "Please wait…" : "Download / resume"}
        </button>
      </div>
    </form>
  </section>;
}
