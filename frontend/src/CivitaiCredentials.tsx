import { useEffect, useState } from "react";

type Status = { configured: boolean; source: string; vault_available: boolean };

export function CivitaiCredentials({ api }: {
  api: <T>(path: string, body?: unknown, signal?: AbortSignal) => Promise<T>;
}) {
  const [status, setStatus] = useState<Status>();
  const [key, setKey] = useState("");
  const [persist, setPersist] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    api<Status>("civitai/credentials", undefined, controller.signal)
      .then(setStatus)
      .catch((reason: Error) => {
        if (reason.name !== "AbortError") setError("Could not read Civitai credential status.");
      });
    return () => controller.abort();
  }, []);
  async function save(clear = false) {
    setBusy(true);
    setError("");
    const input = clear ? "" : key;
    setKey("");
    try {
      setStatus(await api<Status>("civitai/credentials", { key: input, persist: clear ? false : persist }));
    } catch {
      setError("Could not update the Civitai key. Check the key and selected storage.");
    } finally {
      setBusy(false);
    }
  }
  const sources: Record<string, string> = {
    session: "Current server session", environment: "Environment variable",
    windows_vault: "Windows Credential Manager", none: "No key configured",
    unavailable: "Credential Manager could not be read",
  };
  return <section className="conversion-panel" aria-labelledby="civitai-key-title">
    <div className="panel-heading"><h2 id="civitai-key-title">Civitai API key</h2></div>
    <p role="status">{status ? sources[status.source] : "Reading credential status…"}</p>
    <form onSubmit={(event) => { event.preventDefault(); void save(); }}>
      <fieldset disabled={busy || !status}>
        <label className="field-label" htmlFor="civitai-key">Enter or replace API key</label>
        <input id="civitai-key" type="password" value={key}
          onChange={(event) => setKey(event.target.value)} autoComplete="off"
          spellCheck={false} maxLength={1024} placeholder="Civitai API key" />
        {status?.vault_available && <label className="checkbox tool-overwrite">
          <input type="checkbox" checked={persist} onChange={(event) => setPersist(event.target.checked)} />
          Save in Windows Credential Manager
        </label>}
        <p className="preview-note">Otherwise used only until the server restarts. Existing keys are never displayed.
          Clearing the session key restores the environment or stored key.</p>
        <div className="form-actions">
          <button className="secondary" type="button" disabled={status?.source !== "session"}
            onClick={() => void save(true)}>Clear session key</button>
          <button className="primary" type="submit" disabled={!key.trim()}>{busy ? "Saving…" : "Use key"}</button>
        </div>
      </fieldset>
      {error && <p className="picker-error" role="alert">{error}</p>}
    </form>
  </section>;
}
