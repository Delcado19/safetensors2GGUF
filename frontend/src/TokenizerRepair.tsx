import { useState } from "react";

export type TokenizerStatus = {
  status: "missing" | "invalid" | "valid" | "not_checked";
  repairable: boolean;
  message?: string;
};

export function TokenizerRepair({ source, check, busy, repair }: {
  source: string;
  check?: TokenizerStatus | null;
  busy: boolean;
  repair: (body: unknown) => Promise<void>;
}) {
  const [tokenizer, setTokenizer] = useState("");
  if (!check || !["missing", "invalid"].includes(check.status)) return null;
  return <div className="encoder-note" role="status">
    <strong>Tokenizer {check.status === "missing" ? "missing" : "invalid"}</strong>
    <p>{check.message}</p>
    {check.repairable && <>
      <label className="field-label">Tokenizer file from the same model repository
        <input value={tokenizer} onChange={event => setTokenizer(event.target.value)}
          placeholder="C:\models\tokenizer.json" disabled={busy} spellCheck={false} />
      </label>
      <button type="button" className="secondary" disabled={busy || !tokenizer.trim()}
        onClick={() => repair({ operation: "tokenizer_repair", source, tokenizer: tokenizer.trim() })}>
        Create repaired copy
      </button>
      <p className="preview-note">Creates a separate -tokenizer-fixed.safetensors file. Original weights remain unchanged.</p>
    </>}
  </div>;
}
