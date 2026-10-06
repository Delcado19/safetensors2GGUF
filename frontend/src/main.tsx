import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  ArrowDownToLine,
  Box,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Clock3,
  FileBox,
  Folder,
  FolderOpen,
  Layers3,
  Moon,
  Play,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Square,
  Sun,
  X,
} from "lucide-react";
import "./styles.css";
import { ToolForm } from "./ToolForm";
import { HuggingFaceForm } from "./HuggingFaceForm";
import { SupportMatrix } from "./SupportMatrix";
import { FormatSelect, formatName } from "./FormatSelect";

type Container = "gguf" | "safetensors";
type Config = {
  token: string;
  formats: Record<Container, [string, string][]>;
  text_encoder_formats: Record<Container, [string, string][]>;
  executable: string;
  home: string;
  platform: string;
  hf_authenticated: boolean;
};
type Job = {
  id: string;
  status: string;
  progress: number;
  phase: string;
  output: string;
  error: string;
  logs: string[];
  source: string;
  format: string;
  started: number;
  finished: number | null;
  model_kind: "diffusion" | "text_encoder";
  indeterminate: boolean;
  operation: string;
  outputs: string[];
  result:
    | {
        name: string;
        status: string;
        output_tensors: number;
        exact_matches: number;
        mismatches: number;
        reference_path: string | null;
        component_hash?: string;
        reference_hash?: string | null;
        action?: string;
        reference_error?: string | null;
      }[]
    | null;
};
type Inspection = {
  architecture: string;
  source_bytes: number;
  estimated_bytes: number | null;
  destination: string;
  support?: string;
  support_reason?: string | null;
  formats?: [string, string][];
  backend?: string;
};
type Listing = {
  path: string;
  parent: string;
  truncated: boolean;
  entries: { name: string; path: string; directory: boolean; size: number }[];
};
type Location = { label: string; path: string; group: string };
const terminal = (job: Job) =>
  ["succeeded", "failed", "cancelled"].includes(job.status);
const bytes = (value: number) =>
  value >= 2 ** 30
    ? `${(value / 2 ** 30).toFixed(2)} GiB`
    : value >= 2 ** 20
      ? `${(value / 2 ** 20).toFixed(1)} MiB`
      : `${(value / 1024).toFixed(1)} KiB`;
const jobLabel = (job: Job) =>
  (
    ({
      components: "Component export",
      analyze: "Component comparison",
      pad_tokens: "Pad-token repair",
      restore_5d: "5D restoration",
      download: "Hub download",
    }) as Record<string, string>
  )[job.operation] || formatName(job.format);
const filename = (path: string) => path.split(/[\\/]/).pop() || path;

function App() {
  const [config, setConfig] = useState<Config>();
  const [error, setError] = useState("");
  const [view, setView] = useState<
    "convert" | "tools" | "download" | "activity" | "guide"
  >("convert");
  const [light, setLight] = useState(
    () => localStorage.getItem("workbench-theme") === "light",
  );
  const [source, setSource] = useState("");
  const [modelKind, setModelKind] = useState<"diffusion" | "text_encoder">(
    "diffusion",
  );
  const [baseRepo, setBaseRepo] = useState("");
  const [destination, setDestination] = useState("");
  const [toolSource, setToolSource] = useState("");
  const [toolDestination, setToolDestination] = useState("");
  const [downloadDestination, setDownloadDestination] = useState("");
  const [supportSelection, setSupportSelection] = useState("");
  const [container, setContainer] = useState<Container>("gguf");
  const [format, setFormat] = useState("Q4_K_M");
  const [profile, setProfile] = useState("auto");
  const [executable, setExecutable] = useState("");
  const [threads, setThreads] = useState(0);
  const [overwrite, setOverwrite] = useState(false);
  const [keepIntermediate, setKeepIntermediate] = useState(false);
  const [inspection, setInspection] = useState<Inspection>();
  const [sourceFormats, setSourceFormats] = useState<{
    source: string;
    choices: [string, string][];
    backend?: string;
  }>();
  const kreaSource =
    modelKind === "diffusion" && container === "gguf" &&
    sourceFormats?.source === source && sourceFormats.backend === "molbal";
  const [inspecting, setInspecting] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<Job>();
  const [history, setHistory] = useState<Job[]>([]);
  const [picker, setPicker] = useState<"source" | "destination" | null>(null);
  const [listing, setListing] = useState<Listing>();
  const [locations, setLocations] = useState<Location[]>([]);
  const [folderPath, setFolderPath] = useState("");
  const [browsing, setBrowsing] = useState(false);
  const [pickerError, setPickerError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const browseVersion = useRef(0);
  const inspectVersion = useRef(0);
  const active = !!job && !terminal(job);
  const busy = active || history.some((item) => !terminal(item));

  async function api<T>(
    path: string,
    body?: unknown,
    signal?: AbortSignal,
  ): Promise<T> {
    const response = await fetch(`/api/${path}`, {
      method: body === undefined ? "GET" : "POST",
      signal,
      headers: {
        "Content-Type": "application/json",
        "X-Workbench-Token": config?.token || "",
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(
        typeof result.detail === "string"
          ? result.detail
          : "Please check your input.",
      );
    }
    return response.json();
  }

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/config", { signal: controller.signal })
      .then((response) => {
        if (!response.ok)
          throw new Error("Could not connect to the workbench.");
        return response.json();
      })
      .then((result: Config) => {
        setConfig(result);
        setExecutable(result.executable);
      })
      .catch((reason) => {
        if (reason.name !== "AbortError") setError(reason.message);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = light ? "light" : "dark";
    localStorage.setItem("workbench-theme", light ? "light" : "dark");
  }, [light]);

  // Changes invalidate estimates immediately; no stale response may replace them.
  useEffect(() => {
    // Matrix evidence for the previous model cannot follow a replacement source.
    setSupportSelection("");
  }, [source]);

  useEffect(() => {
    inspectVersion.current++;
    setInspection(undefined);
    setInspecting(false);
  }, [
    source,
    destination,
    format,
    container,
    profile,
    overwrite,
    modelKind,
    baseRepo,
  ]);

  useEffect(() => {
    if (!config) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const jobs = await api<Job[]>("jobs", undefined, controller.signal);
        setHistory(jobs);
        setJob((current) =>
          current
            ? jobs.find((item) => item.id === current.id) || current
            : jobs.find((item) => !terminal(item)),
        );
      } catch (reason) {
        if (!controller.signal.aborted) setError((reason as Error).message);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 800);
      }
    };
    poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [config]);

  useEffect(() => {
    if (picker) dialog.current?.showModal();
    else dialog.current?.close();
  }, [picker]);

  useEffect(() => {
    if (!picker) return;
    const controller = new AbortController();
    api<Location[]>("locations", undefined, controller.signal)
      .then(setLocations)
      .catch((reason) => {
        if (!controller.signal.aborted) setPickerError(reason.message);
      });
    return () => controller.abort();
  }, [picker]);

  function parameters() {
    return {
      source,
      model_kind: modelKind,
      base_repo_id:
        modelKind === "text_encoder" && container === "gguf" ? baseRepo : "",
      destination,
      container,
      format,
      precision_profile: modelKind === "diffusion" && !kreaSource ? profile : "auto",
      executable: modelKind === "diffusion" && !kreaSource ? executable : "",
      threads: modelKind === "diffusion" && !kreaSource ? threads : 0,
      overwrite,
      keep_intermediate: modelKind === "diffusion" && !kreaSource && keepIntermediate,
    };
  }

  async function inspect() {
    const version = ++inspectVersion.current;
    setInspecting(true);
    setError("");
    try {
      const result = await api<Inspection>("inspect", parameters());
      if (version === inspectVersion.current) {
        setInspection(result);
        if (result.formats) setSourceFormats({ source, choices: result.formats, backend: result.backend });
      }
    } catch (reason) {
      if (version === inspectVersion.current)
        setError((reason as Error).message);
    } finally {
      if (version === inspectVersion.current) setInspecting(false);
    }
  }

  async function start(body?: unknown, endpoint = "tools") {
    setSubmitting(true);
    setError("");
    try {
      setJob(await api<Job>(body ? endpoint : "jobs", body || parameters()));
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  async function browse(path: string, target = picker) {
    const version = ++browseVersion.current;
    setFolderPath(path);
    setBrowsing(true);
    setPickerError("");
    try {
      const result = await api<Listing>(
        `files?path=${encodeURIComponent(path)}&directories_only=${target === "destination"}`,
      );
      if (version === browseVersion.current) {
        setListing(result);
        // Preserve typing while the directory request is in flight.
        setFolderPath((current) => (current === path ? result.path : current));
      }
    } catch (reason) {
      if (version === browseVersion.current)
        setPickerError((reason as Error).message);
    } finally {
      if (version === browseVersion.current) setBrowsing(false);
    }
  }

  function openPicker(target: "source" | "destination") {
    setPicker(target);
    setListing(undefined);
    const current =
      view === "download"
        ? downloadDestination
        : view === "tools"
          ? target === "source"
            ? toolSource
            : toolDestination
          : target === "source"
            ? source
            : destination;
    const parent = current.replace(/[\\/][^\\/]*$/, "");
    browse(parent && parent !== current ? parent : config?.home || "", target);
  }

  const saving =
    inspection?.estimated_bytes != null
      ? 1 - inspection.estimated_bytes / inspection.source_bytes
      : null;
  const statusText = !job
    ? "Ready when you are"
    : job.status === "succeeded"
      ? job.operation === "analyze"
        ? "Comparison complete"
        : "Your model is ready"
      : job.status === "failed"
        ? "Something needs attention"
        : job.status === "cancelled"
          ? "Job cancelled"
          : job.phase;
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside className="sidebar">
        <a
          href="#"
          className="brand"
          onClick={(event) => {
            event.preventDefault();
            setView("convert");
          }}
          aria-label="safetensors home"
        >
          <span className="brand-mark">
            <Layers3 size={23} />
          </span>
          <span>
            safetensors<span className="brand-sub">MODEL WORKBENCH</span>
          </span>
        </a>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          <button
            className={view === "convert" ? "nav-item selected" : "nav-item"}
            onClick={() => setView("convert")}
            aria-current={view === "convert" ? "page" : undefined}
          >
            <Box size={18} />
            Convert model
            <ChevronRight size={15} />
          </button>
          <button
            className={view === "activity" ? "nav-item selected" : "nav-item"}
            onClick={() => setView("activity")}
            aria-current={view === "activity" ? "page" : undefined}
          >
            <Clock3 size={18} />
            Activity{active && <span className="activity-dot" />}
          </button>
          <button
            className={view === "tools" ? "nav-item selected" : "nav-item"}
            onClick={() => setView("tools")}
            aria-current={view === "tools" ? "page" : undefined}
          >
            <Settings2 size={18} />
            Extract & repair
          </button>
          <button
            className={view === "guide" ? "nav-item selected" : "nav-item"}
            onClick={() => setView("guide")}
            aria-current={view === "guide" ? "page" : undefined}
          >
            <CircleHelp size={18} />
            Format guide
          </button>
          <button
            className={view === "download" ? "nav-item selected" : "nav-item"}
            onClick={() => setView("download")}
            aria-current={view === "download" ? "page" : undefined}
          >
            <ArrowDownToLine size={18} />
            Hugging Face
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="local-note">
            <ShieldCheck size={18} />
            <div>
              Local by design<span>Your models stay on this machine.</span>
            </div>
          </div>
          <button className="theme-button" onClick={() => setLight(!light)}>
            {light ? <Moon size={17} /> : <Sun size={17} />}{" "}
            {light ? "Dark appearance" : "Light appearance"}
          </button>
          <span className="version">safetensors2GGUF · Preview</span>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <span>
            Workspace <ChevronRight size={13} />{" "}
            <strong>
              {view === "convert"
                ? "Convert model"
                : view === "download"
                  ? "Hugging Face"
                  : view === "tools"
                    ? "Extract & repair"
                    : view === "activity"
                      ? "Activity"
                      : "Format guide"}
            </strong>
          </span>
          <span className="connection">
            <span className={config ? "online-dot" : "offline-dot"} />
            {config ? "Local engine connected" : "Connecting to engine"}
          </span>
        </header>
        <main id="main" tabIndex={-1} className={view === "guide" ? "wide-guide" : undefined}>
          <div className="page-heading">
            <div>
              <span className="eyebrow">LESS WEIGHT. MORE POSSIBILITY.</span>
              <h1>
                {view === "convert"
                  ? "Make room for bigger ideas."
                  : view === "download"
                    ? "Your next model starts here."
                    : view === "tools"
                      ? "Bring every piece into place."
                      : view === "activity"
                        ? "Every conversion, in view."
                        : "Find the right balance."}
              </h1>
              <p>
                {view === "convert"
                  ? "Bring your model. Choose a format. Keep creating."
                  : view === "download"
                    ? "Choose a checkpoint from the Hub. Bring every shard together."
                    : view === "tools"
                      ? "Separate checkpoint components. Restore your GGUF. Keep the original."
                      : view === "activity"
                        ? "Your recent jobs, progress, and results in this session."
                        : "Storage, precision, and compatibility — without the guesswork."}
              </p>
            </div>
            <span className="heading-symbol" aria-hidden="true">
              <Layers3 size={52} strokeWidth={1} />
            </span>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <CircleHelp size={18} />
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={17} />
              </button>
            </div>
          )}
          {view === "convert" && supportSelection && (
            <p className="encoder-note" role="status">
              {supportSelection} Choose a matching source model before
              converting.
            </p>
          )}
          {(view === "convert" || view === "tools" || view === "download") && (
            <div className="work-grid">
              {view === "download" ? (
                <HuggingFaceForm
                  api={api}
                  destination={downloadDestination}
                  setDestination={setDownloadDestination}
                  browse={() => openPicker("destination")}
                  disabled={busy || submitting || !config}
                  authenticated={!!config?.hf_authenticated}
                  submit={(body) => start(body, "hf/download")}
                  bytes={bytes}
                />
              ) : view === "tools" ? (
                <ToolForm
                  source={toolSource}
                  destination={toolDestination}
                  setSource={setToolSource}
                  setDestination={setToolDestination}
                  browse={openPicker}
                  formats={config?.formats}
                  disabled={busy || submitting || !config}
                  submit={start}
                />
              ) : (
                <section
                  className="conversion-panel"
                  aria-labelledby="conversion-title"
                >
                  <div className="panel-heading">
                    <h2 id="conversion-title">New conversion</h2>
                    <span className="tag">
                      {modelKind === "diffusion"
                        ? "DIFFUSION MODEL"
                        : "TEXT ENCODER"}
                    </span>
                  </div>
                  <form
                    onSubmit={(event) => {
                      event.preventDefault();
                      start();
                    }}
                  >
                    <fieldset disabled={busy || submitting}>
                      <legend className="section-label">
                        <span>01</span>Source model
                      </legend>
                      <label className="field-label" htmlFor="model-kind">
                        Model type
                      </label>
                      <select
                        id="model-kind"
                        value={modelKind}
                        onChange={(event) => {
                          const kind = event.target.value as
                            "diffusion" | "text_encoder";
                          setModelKind(kind);
                          setFormat(
                            container === "gguf"
                              ? kind === "text_encoder"
                                ? "F16"
                                : "Q4_K_M"
                              : "FP8_MIXED",
                          );
                        }}
                      >
                        <option value="diffusion">Diffusion model</option>
                        <option value="text_encoder">Text encoder</option>
                      </select>
                      <div
                        className={"source-box" + (source ? " has-source" : "")}
                      >
                        <FileBox size={32} strokeWidth={1.4} />
                        <div>
                          <strong>
                            {source
                              ? filename(source)
                              : "Give your model a new shape"}
                          </strong>
                          <p>
                            {source
                              ? "Local file selected · no upload needed"
                              : "Select a safetensors or checkpoint file from your computer."}
                          </p>
                        </div>
                        <button
                          type="button"
                          className="secondary small"
                          disabled={!config}
                          onClick={() => openPicker("source")}
                        >
                          <FolderOpen size={16} />
                          Browse files
                        </button>
                      </div>
                      <label className="field-label" htmlFor="source">
                        Source path
                      </label>
                      <input
                        id="source"
                        required
                        value={source}
                        onChange={(event) => setSource(event.target.value)}
                        placeholder="/models/your-model.safetensors"
                        spellCheck={false}
                      />
                    </fieldset>
                    <fieldset disabled={busy || submitting}>
                      <legend className="section-label">
                        <span>02</span>Output & precision
                      </legend>
                      <div
                        className="container-options"
                        role="group"
                        aria-label="Output container"
                      >
                        <button
                          type="button"
                          aria-pressed={container === "gguf"}
                          className={
                            container === "gguf"
                              ? "container-option chosen"
                              : "container-option"
                          }
                          onClick={() => {
                            setContainer("gguf");
                            setFormat(
                              modelKind === "text_encoder" ? "F16" : "Q4_K_M",
                            );
                          }}
                        >
                          <Box size={20} />
                          <strong>GGUF</strong>
                          <span>
                            {modelKind === "text_encoder"
                              ? "CLIPLoaderGGUF"
                              : "For ComfyUI-GGUF"}
                          </span>
                          {container === "gguf" && <Check size={15} />}
                        </button>
                        <button
                          type="button"
                          aria-pressed={container === "safetensors"}
                          className={
                            container === "safetensors"
                              ? "container-option chosen"
                              : "container-option"
                          }
                          onClick={() => {
                            setContainer("safetensors");
                            setFormat("FP8_MIXED");
                          }}
                        >
                          <Layers3 size={20} />
                          <strong>Safetensors</strong>
                          <span>
                            {modelKind === "text_encoder"
                              ? "Native CLIPLoader"
                              : "Native ComfyUI loading"}
                          </span>
                          {container === "safetensors" && <Check size={15} />}
                        </button>
                      </div>
                      <label className="field-label" htmlFor="format">
                        Quantization
                      </label>
                      <FormatSelect
                        id="format"
                        value={format}
                        onChange={setFormat}
                        container={container}
                        choices={
                          ((modelKind === "text_encoder"
                            ? config?.text_encoder_formats[container]
                            : container === "gguf" && sourceFormats?.source === source
                              ? sourceFormats.choices
                              : config?.formats[container]) || [
                            [format, format],
                          ]) as [string, string][]
                        }
                      />
                      {kreaSource && (
                        <p className="preview-note">
                          Krea GGUF requires a Krea-capable ComfyUI core and the molbal GGUF loader.
                          Q4_0 was tested on one checkpoint with visible drift; other combinations remain untested. Choose an available format explicitly;
                          size estimates are unavailable for this backend.
                        </p>
                      )}
                      {modelKind === "text_encoder" && (
                        <div className="encoder-note">
                          <strong>
                            {container === "gguf"
                              ? "Original base model, correct tokenizer."
                              : "Local conversion, original tensor names."}
                          </strong>
                          <p>
                            {container === "gguf"
                              ? "Auto-detected families use bundled tokenizer files. First use may download llama.cpp; K-quants need CMake and a C++ compiler. CLIP-L/bigG require safetensors."
                              : "No tokenizer download or llama.cpp needed. Use ComfyUI’s native CLIPLoader. Compatibility depends on the encoder family; inspect before converting."}
                          </p>
                        </div>
                      )}
                      {container === "safetensors" &&
                        modelKind === "diffusion" && (
                          <>
                            <label className="field-label" htmlFor="profile">
                              Precision profile
                            </label>
                            <select
                              id="profile"
                              value={profile}
                              onChange={(event) =>
                                setProfile(event.target.value)
                              }
                            >
                              <option value="auto">
                                Automatic · source-aware
                              </option>
                              <option value="conservative">Conservative</option>
                              <option value="qwen_edit_2511">
                                Qwen Image Edit 2511
                              </option>
                              <option value="z_image_turbo">
                                Z-Image Turbo · experimental
                              </option>
                            </select>
                          </>
                        )}
                      <div className="field-row">
                        <label className="field-label" htmlFor="destination">
                          Save to
                        </label>
                        <button
                          type="button"
                          className="text-button"
                          disabled={!config}
                          onClick={() => openPicker("destination")}
                        >
                          <Folder size={14} />
                          Choose folder
                        </button>
                      </div>
                      <input
                        id="destination"
                        value={destination}
                        onChange={(event) => setDestination(event.target.value)}
                        placeholder="Next to source · automatic filename"
                        spellCheck={false}
                      />
                      <p className="field-hint">
                        A file path or folder. Your original model is preserved.
                      </p>
                      <details className="advanced">
                        <summary>
                          <Settings2 size={16} />
                          Advanced settings
                          <ChevronRight size={15} />
                        </summary>
                        <div className="advanced-content">
                          {container === "gguf" &&
                            modelKind === "text_encoder" && (
                              <>
                                <label
                                  className="field-label"
                                  htmlFor="base-repo"
                                >
                                  Original base model repo ID · optional
                                </label>
                                <input
                                  id="base-repo"
                                  value={baseRepo}
                                  onChange={(event) =>
                                    setBaseRepo(event.target.value)
                                  }
                                  placeholder="Automatic · e.g. Qwen/Qwen3-8B"
                                  spellCheck={false}
                                />
                                <p className="field-hint">
                                  Use the original base model, not the fine-tune
                                  repository. An unbundled repo downloads config
                                  and tokenizer files.
                                </p>
                              </>
                            )}
                          {container === "gguf" &&
                            modelKind === "diffusion" && !kreaSource && (
                              <>
                                <label
                                  className="field-label"
                                  htmlFor="executable"
                                >
                                  llama-quantize executable
                                </label>
                                <input
                                  id="executable"
                                  value={executable}
                                  onChange={(event) =>
                                    setExecutable(event.target.value)
                                  }
                                  placeholder="Auto-detected when available"
                                />
                                <label
                                  className="field-label"
                                  htmlFor="threads"
                                >
                                  CPU threads · 0 means automatic
                                </label>
                                <input
                                  id="threads"
                                  type="number"
                                  min="0"
                                  max="1024"
                                  value={threads}
                                  onChange={(event) =>
                                    setThreads(Number(event.target.value))
                                  }
                                />
                                <label className="checkbox">
                                  <input
                                    type="checkbox"
                                    checked={keepIntermediate}
                                    onChange={(event) =>
                                      setKeepIntermediate(event.target.checked)
                                    }
                                  />
                                  Keep intermediate F16 file
                                </label>
                              </>
                            )}
                          <label className="checkbox">
                            <input
                              type="checkbox"
                              checked={overwrite}
                              onChange={(event) =>
                                setOverwrite(event.target.checked)
                              }
                            />
                            Replace an existing output file
                          </label>
                        </div>
                      </details>
                    </fieldset>
                    <div className="form-actions">
                      <span>
                        <ShieldCheck size={15} />
                        Original stays untouched
                      </span>
                      <button
                        className="primary"
                        disabled={
                          !source.trim() || !config || busy || submitting ||
                          (modelKind === "diffusion" && container === "gguf" && sourceFormats?.source === source && !sourceFormats.choices.some(([, key]) => key === format))
                        }
                        type="submit"
                      >
                        {submitting
                          ? "Starting…"
                          : busy
                            ? "Conversion running"
                            : "Convert model"}
                        <ArrowRight size={17} />
                      </button>
                    </div>
                  </form>
                </section>
              )}
              <aside className="inspector">
                {view === "convert" && (
                  <section className="preview-panel">
                    <div className="panel-heading">
                      <h2>At a glance</h2>
                      <span className="tiny-label">OUTPUT PREVIEW</span>
                    </div>
                    <div className="preview-art" aria-hidden="true">
                      <div className="model-cube source-cube">
                        <Layers3 size={36} />
                      </div>
                      <ArrowRight size={19} />
                      <div className="model-cube output-cube">
                        <Box size={30} />
                      </div>
                      <div className="art-labels">
                        <span>Original</span>
                        <span>
                          {container === "gguf" ? "GGUF" : "Safetensors"}
                        </span>
                      </div>
                    </div>
                    <dl className="preview-details">
                      <div>
                        <dt>Format</dt>
                        <dd>{formatName(format)}</dd>
                      </div>
                      <div>
                        <dt>Architecture</dt>
                        <dd>
                          {inspection?.architecture || "Not inspected yet"}
                        </dd>
                      </div>
                      <div>
                        <dt>Source size</dt>
                        <dd>
                          {inspection ? bytes(inspection.source_bytes) : "—"}
                        </dd>
                      </div>
                      <div>
                        <dt>Estimated output</dt>
                        <dd>
                          {inspection?.estimated_bytes != null
                            ? bytes(inspection.estimated_bytes)
                            : "—"}
                        </dd>
                      </div>
                    </dl>
                    {inspection?.support && (
                      <p className="encoder-note" role="status">
                        <strong>
                          ComfyUI compatibility · {inspection.support}
                        </strong>
                        <span>
                          {inspection.support_reason ||
                            (inspection.support === "unknown"
                              ? "This family/format has no confirmed render result. Validate it in your workflow."
                              : "Based on the project’s existing compatibility evidence.")}
                        </span>
                      </p>
                    )}
                    {saving != null && (
                      <div
                        className={saving > 0.05 ? "saving" : "saving caution"}
                      >
                        <ArrowDown size={16} />
                        {saving > 0
                          ? `${(saving * 100).toFixed(1)}% estimated storage saved`
                          : "No estimated storage saving"}
                      </div>
                    )}
                    <button
                      className="secondary inspect-button"
                      disabled={
                        !source.trim() ||
                        !config ||
                        inspecting ||
                        busy ||
                        submitting
                      }
                      onClick={inspect}
                    >
                      <RefreshCw size={15} />
                      {inspecting
                        ? "Inspecting model…"
                        : inspection
                          ? "Refresh estimate"
                          : "Inspect & estimate"}
                    </button>
                    <p className="preview-note">
                      {modelKind === "text_encoder" &&
                        container === "gguf" &&
                        "GGUF size estimation is unavailable for this encoder path. "}
                      Estimates depend on source precision and retained tensors.
                      Actual file size can differ.
                    </p>
                  </section>
                )}
                <section
                  className={"job-panel " + (job?.status || "")}
                  aria-label="Job status"
                >
                  <div className="job-icon">
                    {job?.status === "succeeded" ? (
                      <CheckCircle2 size={22} />
                    ) : active ? (
                      <Play size={20} />
                    ) : (
                      <Clock3 size={21} />
                    )}
                  </div>
                  <h2 aria-live="polite">{statusText}</h2>
                  <p>
                    {!job
                      ? "Progress and your result will appear here."
                      : job.status === "succeeded"
                        ? job.operation === "analyze"
                          ? "Compared with local references. No files changed."
                          : "Published safely. Ready for your next workflow."
                        : active
                          ? "Working in the background. You can keep browsing."
                          : job.error || "Your original model is safe."}
                  </p>
                  {job && (
                    <>
                      <div
                        className="progress-track"
                        role="progressbar"
                        aria-label="Job progress"
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={
                          job.indeterminate
                            ? undefined
                            : Math.round(job.progress * 100)
                        }
                        aria-valuetext={
                          job.indeterminate
                            ? "Working; see technical log"
                            : undefined
                        }
                      >
                        <span
                          style={{ transform: `scaleX(${job.progress})` }}
                        />
                      </div>
                      <div className="progress-caption">
                        <span>{jobLabel(job)}</span>
                        <strong>
                          {job.indeterminate
                            ? "Working"
                            : `${Math.round(job.progress * 100)}%`}
                        </strong>
                      </div>
                      {job.output && (
                        <div className="output-path">
                          <Check size={14} />
                          <span>{job.output}</span>
                        </div>
                      )}
                      {job.result && (
                        <div className="component-results">
                          {job.result.map((item) => (
                            <div key={item.name}>
                              <strong>
                                {item.name.replace("_", "-").toUpperCase()}
                              </strong>
                              <span>{item.status}</span>
                              {item.action && <small>{item.action}</small>}
                              {item.reference_path && (
                                <small className="output-path">
                                  {item.reference_path}
                                </small>
                              )}
                              {item.component_hash && (
                                <details>
                                  <summary>Comparison details</summary>
                                  <small>{item.output_tensors} exportable tensors · {item.exact_matches} exact · {item.mismatches} different</small>
                                  <small className="output-path">Component SHA-256: {item.component_hash}</small>
                                  {item.reference_hash && <small className="output-path">Reference SHA-256: {item.reference_hash}</small>}
                                  <small>Hashes include normalized tensor names, shapes, datatypes and contents.</small>
                                  {item.reference_error && <small>{item.reference_error}</small>}
                                </details>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                      {job.status !== "succeeded" &&
                        job.outputs?.map((path) => (
                          <div className="output-path" key={path}>
                            <Check size={14} />
                            <span>{path}</span>
                          </div>
                        ))}
                      {active && (
                        <button
                          className="secondary cancel-button"
                          disabled={job.status === "cancelling"}
                          onClick={async () => {
                            try {
                              setJob(
                                await api<Job>(`jobs/${job.id}/cancel`, {}),
                              );
                            } catch (reason) {
                              setError((reason as Error).message);
                            }
                          }}
                        >
                          <Square size={13} />
                          {job.status === "cancelling"
                            ? "Stopping safely…"
                            : "Cancel job"}
                        </button>
                      )}
                      <details className="job-log">
                        <summary>View technical log</summary>
                        <pre>
                          {job.logs.join("\n") ||
                            "Waiting for the first update…"}
                        </pre>
                      </details>
                    </>
                  )}
                </section>
              </aside>
            </div>
          )}
          {view === "activity" && (
            <section className="activity-panel">
              <div className="panel-heading">
                <h2>Recent jobs</h2>
                <span className="tiny-label">THIS SESSION</span>
              </div>
              {!history.length ? (
                <div className="empty-state">
                  <Clock3 size={38} strokeWidth={1.2} />
                  <h2>A fresh start.</h2>
                  <p>Your conversions will appear here once you start a job.</p>
                  <button
                    className="secondary"
                    onClick={() => setView("convert")}
                  >
                    Convert your first model
                    <ArrowRight size={16} />
                  </button>
                </div>
              ) : (
                history.map((item) => (
                  <button
                    className="history-item"
                    key={item.id}
                    onClick={() => {
                      setJob(item);
                      setView(
                        item.operation === "download"
                          ? "download"
                          : item.operation === "convert"
                            ? "convert"
                            : "tools",
                      );
                    }}
                  >
                    <FileBox size={23} />
                    <span>
                      <strong>{filename(item.source)}</strong>
                      <small>
                        {jobLabel(item)} ·{" "}
                        {new Date(item.started * 1000).toLocaleTimeString()}
                      </small>
                    </span>
                    <span className={"status-tag " + item.status}>
                      {item.status}
                    </span>
                    <ChevronRight size={17} />
                  </button>
                ))
              )}
              <p className="session-note">
                History is kept in memory. Restarting the engine clears this
                list; output files remain.
              </p>
            </section>
          )}
          {view === "guide" && (
            <>
              {config && (
                <SupportMatrix
                  api={api}
                  busy={busy || submitting}
                  choose={(
                    kind,
                    key,
                    name,
                    classification,
                    precisionProfile,
                    suggestedFormat,
                  ) => {
                    // GGUF groups all precisions; encoder F16 means safetensors.
                    const target =
                      suggestedFormat || (key === "GGUF"
                        ? "Q4_K_M"
                        : kind === "text_encoder" && key === "F16"
                          ? "F16_ST"
                          : key);
                    setModelKind(kind);
                    setContainer(key === "GGUF" ? "gguf" : "safetensors");
                    setFormat(target);
                    setDestination("");
                    setBaseRepo("");
                    setProfile(
                      key === "GGUF" ? "auto" : precisionProfile || "auto",
                    );
                    setSupportSelection(
                      `${name} · ${formatName(key)} · ${classification}.`,
                    );
                    setView("convert");
                    requestAnimationFrame(() =>
                      document.getElementById("model-kind")?.focus(),
                    );
                  }}
                />
              )}
              <section className="guide-panel">
                <div className="guide-feature">
                  <Box size={32} />
                  <h2>GGUF</h2>
                  <p>
                    Use with the ComfyUI-GGUF loader. Q4_K_M is a practical
                    starting point; higher-bit formats retain more precision.
                    K-quants require llama-quantize.
                  </p>
                </div>
                <div className="guide-feature">
                  <Layers3 size={32} />
                  <h2>Safetensors</h2>
                  <p>
                    Load with ComfyUI's native model loader. Mixed formats
                    preserve sensitive tensors at higher precision. FP8 and INT8
                    may offer little saving when the source is already FP8.
                  </p>
                </div>
                <div className="guide-callout">
                  <ShieldCheck size={22} />
                  <div>
                    <h2>Compatibility is model-specific.</h2>
                    <p>
                      Lower precision can change images. NVFP4 has specific
                      hardware requirements; INT4 ConvRot needs a compatible
                      native ComfyUI/Kitchen loader. Check the repository's
                      model-support documentation before choosing a format.
                    </p>
                    <a
                      href="https://github.com/Delcado19/safetensors2GGUF#supported-models"
                      target="_blank"
                      rel="noreferrer"
                    >
                      Read model support
                      <ArrowRight size={15} />
                    </a>
                  </div>
                </div>
                <div className="guide-callout">
                  <Settings2 size={22} />
                  <div>
                    <h2>Choose with evidence.</h2>
                    <p>
                      Explore diffusion models and text encoders above. Unknown
                      combinations remain untested; known problem combinations
                      include their available explanation.
                    </p>
                  </div>
                </div>
              </section>
            </>
          )}
          <footer className="page-footer">
            <span>Built for your next creation.</span>
            <span>Local processing · ComfyUI workflows</span>
          </footer>
        </main>
      </div>
      <dialog
        ref={dialog}
        onCancel={() => setPicker(null)}
        onClose={() => setPicker(null)}
        className="file-dialog"
        aria-labelledby="picker-title"
      >
        <div className="dialog-heading">
          <div>
            <span className="eyebrow">ON THIS MACHINE</span>
            <h2 id="picker-title">
              {picker === "destination"
                ? "Choose an output folder"
                : "Choose your source model"}
            </h2>
          </div>
          <button
            aria-label="Close file browser"
            onClick={() => setPicker(null)}
          >
            <X size={20} />
          </button>
        </div>
        <div className="folder-location">
          <label className="field-label" htmlFor="folder-location">Drive or location</label>
          <select
            id="folder-location"
            value={locations
              .filter((location) => {
                let folder = (listing?.path || "").replace(/\\/g, "/");
                let root = location.path.replace(/\\/g, "/").replace(/\/$/, "");
                if (config?.platform === "nt") {
                  folder = folder.toLowerCase();
                  root = root.toLowerCase();
                }
                return folder === root || folder.startsWith(root + "/");
              })
              .sort((a, b) => b.path.length - a.path.length)[0]?.path || ""}
            disabled={browsing || !locations.length}
            onChange={(event) => browse(event.target.value)}
          >
            <option value="" disabled>Choose a drive or folder</option>
            {["Places", "Drives"].map((group) => (
              <optgroup key={group} label={group}>
                {locations.filter((location) => location.group === group).map((location) => (
                  <option key={location.path} value={location.path}>{location.label}</option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
        <form
          className="folder-address"
          onSubmit={(event) => {
            event.preventDefault();
            browse(folderPath);
          }}
        >
          <button
            type="button"
            aria-label="Parent directory"
            disabled={browsing || !listing || listing.path === listing.parent}
            onClick={() => browse(listing!.parent)}
          >
            <ArrowUp size={18} />
          </button>
          <label className="sr-only" htmlFor="folder-path">
            Folder path
          </label>
          <input
            id="folder-path"
            value={folderPath}
            onChange={(event) => setFolderPath(event.target.value)}
            placeholder="Enter a local folder path"
          />
          <button className="secondary" type="submit" disabled={browsing}>
            Go
          </button>
        </form>
        {pickerError && (
          <p role="alert" className="picker-error">
            {pickerError}
          </p>
        )}
        <div className="file-list" aria-busy={browsing}>
          {browsing ? (
            <p className="file-empty">Reading folder…</p>
          ) : (
            listing?.entries.map((entry) => (
              <button
                className="file-entry"
                key={entry.path}
                onClick={() => {
                  if (entry.directory) browse(entry.path);
                  else {
                    (view === "tools" ? setToolSource : setSource)(entry.path);
                    setPicker(null);
                  }
                }}
              >
                {entry.directory ? <Folder size={19} /> : <FileBox size={19} />}
                <span>{entry.name}</span>
                <small>
                  {entry.directory ? (
                    <ChevronRight size={15} />
                  ) : (
                    bytes(entry.size)
                  )}
                </small>
              </button>
            ))
          )}
          {!browsing && listing?.entries.length === 0 && (
            <p className="file-empty">
              {picker === "destination"
                ? "No subfolders. You can select this folder."
                : "No model files in this folder."}
            </p>
          )}
        </div>
        <div className="dialog-footer">
          <span>
            {listing?.truncated
              ? "First 1,000 entries shown. Enter a more specific folder."
              : "Files stay local. Nothing is uploaded."}
          </span>
          {picker === "destination" && (
            <button
              className="primary"
              disabled={!listing || browsing}
              onClick={() => {
                (view === "download"
                  ? setDownloadDestination
                  : view === "tools"
                    ? setToolDestination
                    : setDestination)(listing!.path + "/");
                setPicker(null);
              }}
            >
              Use this folder
              <Check size={16} />
            </button>
          )}
        </div>
      </dialog>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
