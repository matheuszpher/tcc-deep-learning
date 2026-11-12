"""Gera results/bench_frameworks/resumo.md a partir de bench.jsonl e compat.jsonl."""

import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "results" / "bench_frameworks"


def main():
    rows = [json.loads(line) for line in (OUT / "bench.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    cols = ["model", "length", "framework", "status", "params_trainable", "first_epoch_s",
            "epoch_median_s", "peak_gpu_mb", "final_loss"]
    df = df[[c for c in cols if c in df.columns]].sort_values(["length", "model", "framework"])
    lines = ["# Benchmark TensorFlow x PyTorch", "", df.round(4).to_markdown(index=False), ""]
    ok = df[df.status == "ok"].pivot_table(index=["model", "length"], columns="framework",
                                          values="epoch_median_s")
    if {"tensorflow", "pytorch"} <= set(ok.columns):
        ok["razao_tf_sobre_torch"] = ok["tensorflow"] / ok["pytorch"]
        lines += ["## Tempo mediano por época (s)", "", ok.round(4).to_markdown(), ""]
    cj = OUT / "compat.jsonl"
    if cj.exists():
        lines += ["## Compatibilidade", ""]
        for line in cj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                lines.append("- `" + line.strip() + "`")
    (OUT / "resumo.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
