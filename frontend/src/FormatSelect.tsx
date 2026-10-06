/** Display aliases only: API keys, writer metadata and filenames stay unchanged. */
export function formatName(key: string): string {
  const base = key.replace(/_MIXED$/, "");
  const name =
    (
      {
        F16: "FP16",
        F16_ST: "FP16",
        F32: "FP32",
        FP8: "FP8 (E4M3)",
        INT8: "INT8 + ConvRot",
        INT4_CONVROT: "INT4 + ConvRot",
      } as Record<string, string>
    )[base] || base;
  return name + (key.endsWith("_MIXED") ? " · mixed precision" : "");
}

export function FormatSelect({
  id,
  value,
  choices,
  container,
  onChange,
}: {
  id: string;
  value: string;
  choices: [string, string][];
  container: "gguf" | "safetensors";
  onChange: (value: string) => void;
}) {
  const base = value.replace(/_MIXED$/, "");
  const mixed = value.endsWith("_MIXED");
  const description =
    container === "gguf"
      ? ["F16", "F32", "BF16"].includes(value)
        ? `Floating-point GGUF. ${formatName(value)} uses the internal GGUF type ${value}.`
        : /^Q[45]_[01]$/.test(value)
          ? `GGUF ${value}. Krea uses the pinned streaming converter and requires a matching molbal loader. Q4_0 has a scoped render test with visible drift; this does not validate every model or precision.`
          : `GGUF ${value}. Q8_0 is written directly; K-quants require llama-quantize. Bit width alone does not predict visual fidelity.`
      : (
          {
            F16: "Half-precision float; a precision cast rather than low-bit quantization.",
            F16_ST: "Half-precision float; casts text-encoder tensors to FP16.",
            FP8: "Scaled FP8 E4M3 (float8_e4m3fn).",
            INT8: "Native INT8 tensorwise storage, with ConvRot rotation on eligible linear layers.",
            INT4_CONVROT: "Native ConvRot W4A4: packed INT4 weights with FP32 row scales. Requires ComfyUI convrot_w4a4 support and Kitchen TensorCoreConvRotW4A4Layout (tested 0.2.36), NVIDIA SM 7.5+ for native compute. Render evidence is scoped to SDXL and Z-Image Turbo.",
            NVFP4:
              "NVIDIA FP4 with block scaling. Native accelerated compute requires compatible Blackwell hardware.",
          } as Record<string, string>
        )[base] || "Use a matching ComfyUI loader for this output.";
  const groups =
    container === "gguf"
      ? [
          {
            name: "Floating point",
            match: (key: string) => ["F16", "F32", "BF16"].includes(key),
          },
          {
            name: "8-bit quantization",
            match: (key: string) => key === "Q8_0",
          },
          {
            name: "Block quants",
            match: (key: string) => /^Q[45]_[01]$/.test(key),
          },
          {
            name: "K-quants",
            match: (key: string) => key.startsWith("Q") && key.includes("_K"),
          },
        ]
      : [
          {
            name: "Standard policy",
            match: (key: string) => !key.endsWith("_MIXED"),
          },
          {
            name: "Mixed precision · protected weights",
            match: (key: string) => key.endsWith("_MIXED"),
          },
        ];
  return (
    <>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-describedby={id + "-description"}
      >
        {!choices.some(([, key]) => key === value) && (
          <option value={value} disabled>{formatName(value)} — unavailable; choose a format</option>
        )}
        {groups.map((group) => {
          const entries = choices.filter(([, key]) => group.match(key));
          return entries.length ? (
            <optgroup label={group.name} key={group.name}>
              {entries.map(([, key]) => (
                <option key={key} value={key}>
                  {formatName(key)}
                </option>
              ))}
            </optgroup>
          ) : null;
        })}
      </select>
      <p className="preview-note" id={id + "-description"}>
        {description}
        {container === "safetensors" &&
          (mixed
            ? " Mixed precision is this tool’s model-specific protection policy: selected weights retain source precision. It is not a separate bit format or a universal recipe."
            : " Standard policy still retains tensors required for loader compatibility; it does not force every tensor to this dtype.")}
      </p>
    </>
  );
}
