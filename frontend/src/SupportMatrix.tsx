import { useEffect, useState } from "react";
import { formatName } from "./FormatSelect";
import {
  ArrowRight,
  CheckCircle2,
  CircleHelp,
  TriangleAlert,
  XCircle,
} from "lucide-react";

type Level = "verified" | "caution" | "bad" | "unknown";
type Kind = "diffusion" | "text_encoder";
type Row = { display_name: string; arch?: string; family?: string } & Record<
  string,
  string | null | undefined
>;
type Matrix = Record<Kind, { formats: string[]; rows: Row[] }>;
const levels = {
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
        row.display_name.toLowerCase().includes(query.trim().toLowerCase()) &&
        (filter === "all" ||
          data.formats.some((format) => row[format] === filter)),
    ) || [];
  const level = selected?.row[selected.format] as Level | undefined;
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
            {rows.length} model families shown. The evidence filter keeps rows
            with at least one matching cell.
          </p>
          <div
            className="support-scroll"
            tabIndex={0}
            role="region"
            aria-label="Compatibility table; scroll horizontally for all formats"
          >
            <table className="support-table">
              <caption className="sr-only">
                {kind === "diffusion" ? "Diffusion model" : "Text encoder"}{" "}
                compatibility by output format
              </caption>
              <thead>
                <tr>
                  <th scope="col">Model family</th>
                  {data!.formats.map((format) => (
                    <th scope="col" key={format}>
                      {label(format)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.arch || row.family}>
                    <th scope="row">{row.display_name}</th>
                    {data!.formats.map((format) => {
                      const status = row[format] as Level;
                      const info = levels[status];
                      return (
                        <td key={format}>
                          <button
                            className={"support-cell " + status}
                            aria-label={`${row.display_name}, ${label(format)}: ${info.label}`}
                            aria-pressed={
                              selected?.row === row &&
                              selected.format === format
                            }
                            aria-describedby={
                              row[format + "__scope"]
                                ? `${row.arch || row.family}-${format}-scope`
                                : undefined
                            }
                            onClick={() => setSelected({ row, format })}
                          >
                            <info.icon size={18} />
                            <span>{info.label}</span>
                            {row[format + "__scope"] && (
                              <small
                                id={`${row.arch || row.family}-${format}-scope`}
                              >
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
              No matching model families. Try another name or evidence filter.
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
            {levels[level].label} · {label(selected.format)}
          </span>
          <h3>{selected.row.display_name}</h3>
          <p>
            {selected.row[selected.format + "__reason"] || levels[level].note}
          </p>
          {selected.format === "GGUF" && (
            <p className="preview-note">
              GGUF groups multiple precisions. Use format selects Q4_K_M; this
              cell does not verify every quantization level.
            </p>
          )}
          <p className="preview-note">
            Choose a source matching this family. Selection changes form
            settings; it does not inspect or convert a model.
          </p>
          <button
            className="primary"
            disabled={busy || level === "bad"}
            onClick={() =>
              choose(
                kind,
                selected.format,
                selected.row.display_name,
                levels[level].label,
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
        aria-label="INT4 ConvRot prototype status"
      >
        <span className="tiny-label">PROTOTYPE · NOT SELECTABLE</span>
        <h3>INT4 + ConvRot</h3>
        <p>
          Native 4-bit ConvRot has passed SDXL rendering, LoRA and offload tests
          on one checkpoint. It remains outside the standard conversion formats
          and family matrix until version guards and supported-model integration
          are implemented.
        </p>
        <p className="preview-note">
          INT8 + ConvRot is already implemented and appears in the INT8
          columns/options above. ConvRot is a rotation method, not a bit width.
        </p>
      </div>
    </section>
  );
}
