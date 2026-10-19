import { useState } from "react";

export type TokenizerStatus = {
  status: "missing" | "invalid" | "valid" | "not_checked";
  repairable: boolean;
  message?: string;
  repair_kind?: "runtime";
  resource_path?: string;
};

export function TokenizerRepair({ source, check, busy, repair }: {
  source: string;
  check?: TokenizerStatus | null;
  busy: boolean;
  repair: (body: unknown) => Promise<void>;
}) {
  const [tokenizer, setTokenizer] = useState("");
  if (!check) return null;
  return <div className="encoder-note" role="status">
    <strong>Tokenizer {check.status.replace("_", " ")}</strong>
    <p>{check.message}</p>
    {check.resource_path && <p className="preview-note">{check.resource_path}</p>}
    {check.repairable && <>
      <label className="field-label">{check.repair_kind === "runtime" ? "Healthy tokenizer directory from the same ComfyUI version" : "Tokenizer file from the same model repository"}
        <input value={tokenizer} onChange={event => setTokenizer(event.target.value)}
          placeholder="C:\models\tokenizer.json" disabled={busy} spellCheck={false} />
      </label>
      <button type="button" className="secondary" disabled={busy || !tokenizer.trim()}
        onClick={() => repair({ operation: "tokenizer_repair", source, tokenizer: tokenizer.trim() })}>
        {check.repair_kind === "runtime" ? "Repair shared runtime tokenizer" : "Create repaired copy"}
      </button>
      <p className="preview-note">{check.repair_kind === "runtime" ? "Replaces shared Qwen tokenizer assets, retains a backup directory. Restart ComfyUI afterward. Model weights remain unchanged." : "Creates a separate -tokenizer-fixed.safetensors file. Original weights remain unchanged."}</p>
    </>}
  </div>;
}
