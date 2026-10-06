import { useEffect, useState } from "react";
import { formatName } from "./FormatSelect";
import {
  ArrowRight,
  CheckCircle2,
  CircleHelp,
  TriangleAlert,
  XCircle,
  Clock3,
} from "lucide-react";

type Level = "verified" | "caution" | "bad" | "unknown" | "pending";
type Kind = "diffusion" | "text_encoder";
type Row = {
  id: string;
  display_name: string;
  arch?: string;
  family?: string;
  precision_profile?: string;
} & Record<string, string | null | undefined>;
type Matrix = Record<Kind, { formats: string[]; rows: Row[] }>;
const levels = {
  pending: {
    label: "Integration pending",
    icon: Clock3,
    note: "Local integration or validation is pending. This is not a claim that the format is fundamentally unsupported.",
  },
  verified: {
    label: "Verified",
    icon: CheckCircle2,
    note: "The project classifies this combination as supported based on tests or implementation reasoning. It does not cover every model release, quantization level or runtime configuration.",
  },
  caution: {
    label: "Visible drift",
    icon: TriangleAlert,
    note: "Project render tests found visible but tolerable deviations from the baseline. Compare results in your own workflow.",
  },
  bad: {
    label: "Not supported",
    icon: XCircle,
    note: "The project records a tooling limitation or failed render evidence for this combination. Choose a supported alternative.",
  },
  unknown: {
    label: "Untested",
    icon: CircleHelp,
    note: "No confirmed project render evidence for this combination. This is neither a success nor a failure claim.",
  },
};
const label = formatName;
const cellLevel = (row: Row, format: string): Level =>
  row[format + "__pending"] === "true" ? "pending" : (row[format] as Level);
const cellLabel = (row: Row, format: string) =>
  row[format + "__label"] || levels[cellLevel(row, format)].label;
function headerLines(key: string) {
  if (key.endsWith("_MIXED"))
    return [label(key.replace(/_MIXED$/, "")), "mixed precision"];
  if (key === "INT8") return ["INT8", "+ConvRot"];
  if (key === "FP8") return ["FP8", "(E4M3)"];
  return [label(key)];
}

export function SupportMatrix({
  api,
  busy,
  choose,
}: {
  api: <T>(path: string, body?: unknown, signal?: AbortSignal) => Promise<T>;
  busy: boolean;
  choose: (
    kind: Kind,
    format: string,
    name: string,
    classification: string,
    profile?: string,
    suggestedFormat?: string,
  ) => void;
}) {
  const [matrix, setMatrix] = useState<Matrix>();
  const [error, setError] = useState("");
  const [kind, setKind] = useState<Kind>("diffusion");
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Level | "all">("all");
  const [selected, setSelected] = useState<{ row: Row; format: string }>();
  useEffect(() => {
    const controller = new AbortController();
    api<Matrix>("support", undefined, controller.signal)
      .then(setMatrix)
      .catch((reason) => {
        if (!controller.signal.aborted) setError(reason.message);
      });
    return () => controller.abort();
    // Fetch once per mounted guide. Polling job updates must not refetch the matrix.
  }, []);
  const data = matrix?.[kind];
  const rows =
    data?.rows.filter(
      (row) =>
        `${row.display_name} ${row.arch || row.family}`
          .toLowerCase()
          .includes(query.trim().toLowerCase()) &&
        (filter === "all" ||
          data.formats.some((format) => cellLevel(row, format) === filter)),
    ) || [];
  const level = selected ? cellLevel(selected.row, selected.format) : undefined;
  return (
    <section className="support-panel" aria-labelledby="support-title">
      <div className="panel-heading">
        <h2 id="support-title">Compatibility, in view.</h2>
        <span className="tiny-label">PROJECT EVIDENCE</span>
      </div>
      <p className="preview-note">
        Shared classifications from the project’s support registry. Verified
        includes render tests and implementation reasoning; it is not a
        guarantee for every release. Select a cell for details.
      </p>
      <div className="support-filters">
        <div>
          <label className="field-label" htmlFor="support-kind">
            Model category
          </label>
          <select
            id="support-kind"
            value={kind}
            onChange={(event) => {
              setKind(event.target.value as Kind);
              setSelected(undefined);
            }}
          >
            <option value="diffusion">Diffusion models</option>
            <option value="text_encoder">Text encoders</option>
          </select>
        </div>
        <div>
          <label className="field-label" htmlFor="support-search">
            Find a model
          </label>
          <input
            id="support-search"
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Model name or architecture"
          />
        </div>
        <div>
          <label className="field-label" htmlFor="support-filter">
            Evidence filter
          </label>
          <select
            id="support-filter"
            value={filter}
            onChange={(event) => setFilter(event.target.value as typeof filter)}
          >
            <option value="all">All classifications</option>
            {Object.entries(levels).map(([key, item]) => (
              <option key={key} value={key}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="support-legend">
        {Object.entries(levels).map(([key, item]) => (
          <span className={"support-level " + key} key={key}>
            <item.icon size={15} />
            {item.label}
          </span>
        ))}
      </div>
      {error && (
        <p className="picker-error" role="alert">
          {error} Reopen the guide to retry.
        </p>
      )}
      {!matrix && !error && <p role="status">Loading compatibility data…</p>}
      {matrix && (
        <>
          <p className="preview-note" role="status">
            {rows.length} models / variants shown. The evidence filter keeps
            rows with at least one matching cell.
          </p>
          <div
            className="support-scroll"
            tabIndex={0}
            role="region"
            aria-label="Compatibility table; scroll horizontally for all formats"
          >
            <table className="support-table" style={{ minWidth: `${13 + data!.formats.length * 6}rem` }}>
              <caption className="sr-only">
                {kind === "diffusion" ? "Diffusion model" : "Text encoder"}{" "}
                compatibility by output format
              </caption>
              <colgroup>
                <col className="support-model-column" />
                {data!.formats.map((format) => <col key={format} />)}
              </colgroup>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  {data!.formats.map((format) => (
                    <th scope="col" key={format} aria-label={label(format)}>
                      {headerLines(format).map((line, index) => (
                        <span
                          className={index ? "format-subtitle" : "format-title"}
                          key={line}
                        >
                          {line}
                        </span>
                      ))}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id} data-model-id={row.id}>
                    <th scope="row">
                      <span className="model-name">{row.display_name}</span>
                      <small className="model-code">
                        ({row.arch || row.family})
                      </small>
                    </th>
                    {data!.formats.map((format) => {
                      const status = cellLevel(row, format);
                      const info = levels[status];
                      return (
                        <td key={format}>
                          <button
                            className={"support-cell " + status}
                            aria-label={`${row.display_name}, ${label(format)}: ${cellLabel(row, format)}`}
                            aria-pressed={
                              selected?.row === row &&
                              selected.format === format
                            }
                            aria-describedby={
                              row[format + "__scope"]
                                ? `${row.id}-${format}-scope`
                                : undefined
                            }
                            onClick={() => setSelected({ row, format })}
                          >
                            <info.icon size={18} />
                            <span>{cellLabel(row, format)}</span>
                            {row[format + "__scope"] && (
                              <small id={`${row.id}-${format}-scope`}>
                                {row[format + "__scope"]}
                              </small>
                            )}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!rows.length && (
            <p className="empty-state">
              No matching models or variants. Try another name or evidence
              filter.
            </p>
          )}
        </>
      )}
      {selected && level && (
        <div
          className="support-detail"
          role="region"
          aria-label="Selected compatibility details"
        >
          <span className={"support-level " + level}>
            {cellLabel(selected.row, selected.format)} ·{" "}
            {label(selected.format)}
          </span>
          <h3>{selected.row.display_name}</h3>
          <p>
            {selected.row[selected.format + "__reason"] || levels[level].note}
          </p>
          {selected.format === "GGUF" && (
            <p className="preview-note">
              GGUF groups multiple precisions. Use format selects {selected.row.GGUF__format || "Q4_K_M"}; this
              cell does not verify every quantization level.
            </p>
          )}
          <p className="preview-note">
            Choose a source matching this model and variant. Selection changes
            form settings; it does not inspect or convert a model.
          </p>
          {selected.row[selected.format + "__selectable"] === "false" && (
            <p className="preview-note">Prototype evidence only. Conversion is not yet available in the application.</p>
          )}
          <button
            className="primary"
            disabled={busy || level === "bad" || level === "pending" || selected.row[selected.format + "__selectable"] === "false"}
            onClick={() =>
              choose(
                kind,
                selected.format,
                selected.row.display_name,
                cellLabel(selected.row, selected.format),
                selected.row.precision_profile,
                selected.row[selected.format + "__format"] || undefined,
              )
            }
          >
            Use format
            <ArrowRight size={16} />
          </button>
        </div>
      )}
      <p className="preview-note">
        Visible drift is model/checkpoint, profile and workflow dependent; it
        does not mean every model or seed changes in the same way. Select a cell
        for the observed details and test scope.
      </p>
      <p className="preview-note">
        Storage savings depend on source dtype and retained tensors. Inspect
        your source for an estimate. NVFP4 has hardware requirements.
      </p>
      <div
        className="support-detail"
        aria-label="INT4 ConvRot runtime requirements"
      >
        <span className="tiny-label">NATIVE W4A4 · MIXED PRECISION</span>
        <h3>INT4 + ConvRot</h3>
        <p>
          Native 4-bit ConvRot has passed SDXL rendering, LoRA and offload tests
          on one checkpoint, plus Z-Image Turbo single-pass rendering with two LoRAs.
          Conversion is available with native ComfyUI convrot_w4a4 support and
          Kitchen TensorCoreConvRotW4A4Layout (tested 0.2.36), NVIDIA SM 7.5+ for native compute.
          The diffusion matrix records the tested mixed-precision policy and
          leaves other models untested.
        </p>
        <p className="preview-note">
          INT8 + ConvRot is already implemented and appears in the INT8
          columns/options above. ConvRot is a rotation method, not a bit width.
        </p>
      </div>
    </section>
  );
}
