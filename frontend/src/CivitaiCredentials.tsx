import { useEffect, useState } from "react";
import { CheckCircle2 } from "lucide-react";

type Status = { configured: boolean; source: string; vault_available: boolean };

export function CivitaiCredentials({ api, onConfigured, sessionToken }: {
  api: <T>(path: string, body?: unknown, signal?: AbortSignal) => Promise<T>;
  onConfigured?: (configured: boolean | undefined) => void;
  sessionToken?: string;
}) {
  const [status, setStatus] = useState<Status>();
  const [key, setKey] = useState("");
  const [source, setSource] = useState("");
  const [persist, setPersist] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [reading, setReading] = useState(true);
  const [statusError, setStatusError] = useState(false);
  async function readStatus(signal?: AbortSignal) {
    setReading(true);
    setConfirmation("");
    onConfigured?.(undefined);
    setStatusError(false);
    setError("");
    try {
      const result = await api<Status>("civitai/credentials", undefined, signal);
      if (signal?.aborted) return;
      setStatus(result);
      onConfigured?.(result.configured);
    } catch (reason) {
      if (signal?.aborted || (reason as Error).name === "AbortError") return;
      setStatusError(true);
      setError("Could not read Civitai credential status. Retry to continue setup.");
      onConfigured?.(undefined);
    } finally {
      if (!signal?.aborted) setReading(false);
    }
  }
  useEffect(() => {
    const controller = new AbortController();
    void readStatus(controller.signal);
    return () => controller.abort();
  }, [sessionToken]);
  async function save(clear = false) {
    setBusy(true);
    setError("");
    setConfirmation("");
    const input = clear ? "" : key;
    setKey("");
    try {
      const result = await api<Status>("civitai/credentials", {
        key: input, source: clear ? "direct" : source,
        persist: !clear && source === "direct" && persist,
      });
      setStatus(result);
      onConfigured?.(result.configured);
      // Setup confirms local configuration, not access to a particular Civitai model.
      setConfirmation(clear ? "Session key cleared. Automatic credential lookup restored."
        : source === "environment" ? "Environment key loaded for this session."
        : source === "windows_vault" ? "Windows credential loaded for this session."
        : persist ? "Key saved in Windows Credential Manager and ready for this session."
        : "Key configured for this session.");
    } catch (reason) {
      const message = (reason as Error).message;
      setError(message === "Invalid credential request."
        ? "Restart the Workbench server and reload this page to use credential source selection."
        : message || "Could not update the Civitai key. Check the key and selected storage.");
    } finally {
      setBusy(false);
    }
  }
  const sources: Record<string, string> = {
    session: "Current server session", environment: "Environment variable",
    windows_vault: "Windows Credential Manager", none: "No key configured",
    session_environment: "Current session from environment variable",
    session_windows_vault: "Current session from Windows Credential Manager",
    unavailable: "Credential Manager could not be read",
  };
  return <section className="conversion-panel" aria-labelledby="civitai-key-title">
    <div className="panel-heading"><h2 id="civitai-key-title">Civitai API key</h2></div>
    <p className="credential-current">{reading ? "Reading credential status…" : statusError ? "Credential status unavailable" : status ? sources[status.source] : "Credential status unavailable"}</p>
    {statusError && <button id="credential-status-retry" className="secondary" type="button" disabled={reading || busy}
      onClick={() => void readStatus()}>Retry credential status</button>}
    <form onSubmit={(event) => { event.preventDefault(); void save(); }}>
      <fieldset disabled={busy || reading || !status}>
        <label className="field-label" htmlFor="civitai-source-select">Credential source</label>
        <select id="civitai-source-select" value={source} onChange={(event) => {
          setSource(event.target.value); setKey(""); setPersist(false); setError(""); setConfirmation("");
        }}>
          <option value="" disabled>Choose credential source…</option>
          <option value="direct">API key</option>
          <option value="environment">Environment variable</option>
          <option value="windows_vault" disabled={!status?.vault_available}>Windows Credential Manager</option>
        </select>
        <label className="field-label" htmlFor="civitai-key">
          {source === "environment" ? "Variable name" : source === "windows_vault" ? "Credential name" : "API key"}
        </label>
        <input id="civitai-key" type={source === "direct" ? "password" : "text"} value={key} disabled={!source}
          onChange={(event) => { setKey(event.target.value); setConfirmation(""); }} autoComplete="off"
          spellCheck={false} maxLength={1024} placeholder={
            source === "environment" ? "Enter variable name" : source === "windows_vault" ? "Enter credential name" : "Enter API key"
          } />
        {source === "direct" && status?.vault_available && <label className="checkbox tool-overwrite">
          <input type="checkbox" checked={persist} onChange={(event) => setPersist(event.target.checked)} />
          Save in Windows Credential Manager
        </label>}
        <p className="preview-note">Otherwise used only until the server restarts. Existing keys are never displayed.
          Variable and credential names are freely selectable and resolved only on this machine.
          Clearing the session key restores the environment or stored key.</p>
        <div className="form-actions">
          <button className="secondary" type="button" disabled={!status?.source.startsWith("session")}
            onClick={() => void save(true)}>Clear session key</button>
          <button className="primary" type="submit" disabled={!source || !key.trim()}>{busy ? "Saving…" : "Use key"}</button>
        </div>
      </fieldset>
      <div className="credential-feedback" role="status" aria-live="polite" aria-atomic="true">
        {confirmation && <p className="credential-success"><CheckCircle2 size={18} aria-hidden="true" />{confirmation}</p>}
        {error && <p className="picker-error" role="alert">{error}</p>}
      </div>
    </form>
  </section>;
}
