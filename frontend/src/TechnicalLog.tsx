import { useLayoutEffect, useRef } from "react";

/** Follow new output unless the user has scrolled up to read earlier lines. */
export function TechnicalLog({ logs }: { logs: string[] }) {
  const output = useRef<HTMLPreElement>(null);
  const follow = useRef(true);
  const text = logs.join("\n");

  useLayoutEffect(() => {
    if (output.current && follow.current)
      output.current.scrollTop = output.current.scrollHeight;
  }, [text]);

  return (
    <details className="job-log" onToggle={(event) => {
      if (event.currentTarget.open && output.current) {
        follow.current = true;
        output.current.scrollTop = output.current.scrollHeight;
      }
    }}>
      <summary>View technical log</summary>
      <pre ref={output} onScroll={(event) => {
        const element = event.currentTarget;
        follow.current = element.scrollHeight - element.clientHeight - element.scrollTop < 24;
      }}>{text || "Waiting for the first update..."}</pre>
    </details>
  );
}
