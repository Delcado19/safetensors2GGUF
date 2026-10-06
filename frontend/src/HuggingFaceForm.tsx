import { useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  FolderOpen,
  Search,
  ShieldCheck,
} from "lucide-react";

type Repository = {
  repo_id: string;
  revision: string;
  groups: { subfolder: string; files: number; bytes: number | null }[];
};

export function HuggingFaceForm({
  api,
  destination,
  setDestination,
  browse,
  disabled,
  authenticated,
  submit,
  bytes,
}: {
  api: <T>(path: string, body?: unknown, signal?: AbortSignal) => Promise<T>;
  destination: string;
  setDestination: (value: string) => void;
  browse: () => void;
  disabled: boolean;
  authenticated: boolean;
  submit: (parameters: unknown) => void;
  bytes: (value: number) => string;
}) {
  const [repo, setRepo] = useState("");
  const [revision, setRevision] = useState("main");
  const [inspection, setInspection] = useState<Repository>();
  const [subfolder, setSubfolder] = useState<string | null>(null);
  const [overwrite, setOverwrite] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [error, setError] = useState("");
  const version = useRef(0);
  const request = useRef<AbortController | undefined>(undefined);
  useEffect(() => {
    version.current++;
    request.current?.abort();
    setInspection(undefined);
    setSubfolder(null);
    setInspecting(false);
    setError("");
    return () => {
      version.current++;
      request.current?.abort();
    };
  }, [repo, revision]);
  async function inspect() {
    const current = ++version.current;
    request.current?.abort();
    request.current = new AbortController();
    setInspecting(true);
    setError("");
    try {
      const result = await api<Repository>(
        "hf/inspect",
        { source: repo, revision },
        request.current.signal,
      );
      if (current === version.current) {
        setInspection(result);
        setSubfolder(
          result.groups.length === 1 ? result.groups[0].subfolder : null,
        );
      }
    } catch (reason) {
      if (
        current === version.current &&
        (reason as Error).name !== "AbortError"
      )
        setError((reason as Error).message);
    } finally {
      if (current === version.current) setInspecting(false);
    }
  }
  const selected = inspection?.groups.find(
    (group) => group.subfolder === subfolder,
  );
  const outputName =
    selected &&
    `${selected.subfolder.split("/").pop() || inspection!.repo_id.split("/").pop()}.safetensors`;
  return (
    <section className="conversion-panel" aria-labelledby="hf-title">
      <div className="panel-heading">
        <h2 id="hf-title">Download from the Hub</h2>
        <span className="tag">HUGGING FACE</span>
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (inspection && selected)
            submit({
              source: inspection.repo_id,
              revision: inspection.revision,
              subfolder,
              destination,
              overwrite,
            });
        }}
      >
        <fieldset disabled={disabled}>
          <legend className="section-label">
            <span>01</span>Find your checkpoint
          </legend>
          <label className="field-label" htmlFor="hf-repo">
            Repository ID
          </label>
          <input
            id="hf-repo"
            value={repo}
            onChange={(event) => setRepo(event.target.value)}
            placeholder="organization/model-name"
            spellCheck={false}
            autoCapitalize="none"
          />
          <label className="field-label" htmlFor="hf-revision">
            Revision
          </label>
          <input
            id="hf-revision"
            value={revision}
            onChange={(event) => setRevision(event.target.value)}
            placeholder="main, tag or commit"
            spellCheck={false}
          />
          <button
            type="button"
            className="secondary inspect-button"
            disabled={!repo.trim() || !revision.trim() || inspecting}
            onClick={inspect}
          >
            <Search size={16} />
            {inspecting ? "Reading repository…" : "Inspect repository"}
          </button>
          <p className="preview-note">
            Reads file information from Hugging Face. Model weights download
            only when you start.
          </p>
          {error && (
            <p className="picker-error" role="alert">
              {error}
            </p>
          )}
          {inspection && (
            <>
              <label className="field-label" htmlFor="hf-variant">
                Checkpoint folder
              </label>
              <select
                id="hf-variant"
                value={subfolder ?? "__choose__"}
                onChange={(event) => setSubfolder(event.target.value)}
              >
                <option value="__choose__" disabled>
                  Choose one checkpoint folder
                </option>
                {inspection.groups.map((group) => (
                  <option key={group.subfolder} value={group.subfolder}>
                    {group.subfolder || "Repository root"} · {group.files} file
                    {group.files === 1 ? "" : "s"}
                  </option>
                ))}
              </select>
              <p className="preview-note">
                Only the selected folder is merged. Separate complete models in
                one folder require a dedicated model repository.
              </p>
            </>
          )}
        </fieldset>
        <fieldset disabled={disabled}>
          <legend className="section-label">
            <span>02</span>Bring it home
          </legend>
          <label className="field-label" htmlFor="hf-destination">
            Download folder
          </label>
          <div className="path-field">
            <input
              id="hf-destination"
              required
              value={destination}
              onChange={(event) => setDestination(event.target.value)}
              placeholder="Choose a local folder"
              spellCheck={false}
            />
            <button
              type="button"
              className="secondary"
              onClick={browse}
              aria-label="Browse download folder"
            >
              <FolderOpen size={17} />
            </button>
          </div>
          {selected && (
            <div className="encoder-note hf-plan" role="status">
              <strong>
                <ArrowDownToLine size={17} />
                {outputName}
              </strong>
              <span>
                {selected.files} shard{selected.files === 1 ? "" : "s"} ·{" "}
                {selected.bytes == null
                  ? "Download size unavailable"
                  : bytes(selected.bytes) + " download"}
              </span>
              <small>Revision {inspection!.revision.slice(0, 12)}</small>
            </div>
          )}
          <label className="checkbox tool-overwrite">
            <input
              type="checkbox"
              checked={overwrite}
              onChange={(event) => setOverwrite(event.target.checked)}
            />
            Replace an existing merged file
          </label>
          <p className="preview-note">
            Shards merge into one safetensors file. Keep space for both shards
            and the merged output. Cancel takes effect between transfers;
            partial shards stay for a retry.
          </p>
          <p className="encoder-note">
            <strong>
              {authenticated
                ? "Server login available"
                : "Public repositories ready"}
            </strong>
            <span>
              Private or gated models need a server login and any required
              access approval. Credentials stay on this machine.
            </span>
          </p>
        </fieldset>
        <div className="form-actions">
          <span>
            <ShieldCheck size={15} />
            Publish only when complete
          </span>
          <button
            className="primary"
            type="submit"
            disabled={
              disabled || inspecting || !selected || !destination.trim()
            }
          >
            {disabled ? "Please wait…" : "Download checkpoint"}
            <ArrowRight size={17} />
          </button>
        </div>
      </form>
    </section>
  );
}
