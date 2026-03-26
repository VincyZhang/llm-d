#!/usr/bin/env python3
"""
Benchmark runner for PD break-even evaluation.

Wraps `vllm bench serve` and maps its output to the result schema defined in
TEST_CONFIGURATION_SPEC.md section 8.

Usage:
    python benchmark_runner.py \
        --mode baseline \
        --isl 4096 \
        --osl 256 \
        --concurrency 8 \
        --warmup-requests 30 \
        --measured-requests 300 \
        --output-file results/TC-024_baseline.json

Modes:
    - baseline: colocated serving without PD
    - baseline-pd: PD without llm-d (for example, toy-proxy based validation)
    - llm-d-pd: PD with llm-d routing/gateway

Requirements:
    - vllm CLI on PATH: pip install vllm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_BASELINE_PORT = 8084
DEFAULT_PD_PORT = 8085
DEFAULT_LLM_D_PORT = 8086


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PD break-even benchmark runner (vllm bench serve)")

    p.add_argument("--mode", choices=["baseline", "baseline-pd", "llm-d-pd"], required=True)
    p.add_argument("--case-id", default=None, help="Case ID, e.g. TC-001.")
    p.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")

    p.add_argument("--isl", type=int, required=True, help="Input sequence length (tokens).")
    p.add_argument("--osl", type=int, required=True, help="Output sequence length (tokens).")
    p.add_argument(
        "--concurrency", type=int, required=True,
        help="Concurrent requests (used as --max-concurrency for vllm bench serve).",
    )

    p.add_argument("--warmup-requests", type=int, default=30)
    p.add_argument("--measured-requests", type=int, default=300)
    p.add_argument("--max-retries", type=int, default=0)
    p.add_argument("--random-seed", type=int, default=20260325)

    p.add_argument("--port", type=int, default=None,
                   help="Override server port (default: baseline=8084, baseline-pd=8085, llm-d-pd=8086).")
    p.add_argument("--host", default="localhost")
    p.add_argument("--random-prefix-len", type=int, default=0)
    p.add_argument("--backend", default="vllm")

    p.add_argument("--output-file", default=None, help="Path to write JSON result.")
    return p.parse_args()


def resolve_port(args: argparse.Namespace) -> int:
    if args.port is not None:
        return args.port
    if args.mode == "baseline":
        return DEFAULT_BASELINE_PORT
    if args.mode == "baseline-pd":
        return DEFAULT_PD_PORT
    return DEFAULT_LLM_D_PORT


def build_cmd(args: argparse.Namespace, port: int, raw_dir: str, raw_name: str) -> list[str]:
    """Build the vllm bench serve command."""
    total_prompts = args.warmup_requests + args.measured_requests
    return [
        "vllm", "bench", "serve",
        "--backend",            args.backend,
        "--model",              args.model,
        "--dataset-name",       "random",
        "--random-input-len",   str(args.isl),
        "--random-output-len",  str(args.osl),
        "--random-prefix-len",  str(args.random_prefix_len),
        "--num-prompts",        str(total_prompts),
        "--host",               args.host,
        "--port",               str(port),
        "--max-concurrency",    str(args.concurrency),
        "--save-result",
        "--result-dir",         raw_dir,
        "--result-filename",    raw_name,
    ]


def convert(raw: dict, args: argparse.Namespace, port: int) -> dict:
    """
    Map vllm bench serve JSON output to TEST_CONFIGURATION_SPEC section 8 schema.

    vllm bench serve output fields:
            mean_ttft_ms,
      median_ttft_ms, p99_ttft_ms,
            mean_tpot_ms,
      median_tpot_ms, p99_tpot_ms,
      request_throughput  (req/s),
      output_throughput   (token/s),
      successful_requests, total_requests
    """
    def _f(*keys, default=0.0):
        for k in keys:
            if raw.get(k) is not None:
                return float(raw[k])
        return default

    total = int(raw.get("total_requests", args.warmup_requests + args.measured_requests))
    success = int(raw.get("successful_requests", raw.get("completed", total)))

    errors_timeout = int(raw.get("num_timeout_errors", 0))
    errors_http    = int(raw.get("num_http_errors", 0))
    failed_total = int(raw.get("failed", max(total - success, 0)))
    errors_other = max(failed_total - errors_timeout - errors_http, 0)

    return {
        "case_id":      args.case_id or "N/A",
        "mode":         args.mode,
        "isl":          args.isl,
        "osl":          args.osl,
        "concurrency":  args.concurrency,
        "ttft_ms": {
            "avg": round(_f("mean_ttft_ms", "avg_ttft_ms", "average_ttft_ms"), 4),
            "p50": round(_f("median_ttft_ms", "p50_ttft_ms"), 4),
            "p99": round(_f("p99_ttft_ms"), 4),
        },
        "tpot_ms": {
            "avg": round(_f("mean_tpot_ms", "avg_tpot_ms", "average_tpot_ms"), 4),
            "p50": round(_f("median_tpot_ms", "p50_tpot_ms"), 4),
            "p99": round(_f("p99_tpot_ms"), 4),
        },
        "throughput": {
            "req_per_s":   round(_f("request_throughput"), 6),
            "token_per_s": round(_f("output_throughput"), 6),
        },
        "success_rate": round(success / total, 6) if total > 0 else 0.0,
        "errors": {
            "timeout": errors_timeout,
            "http":    errors_http,
            "other":   errors_other,
        },
        "meta": {
            "model":               args.model,
            "port":                port,
            "dataset_name":        "random",
            "warmup_requests":     args.warmup_requests,
            "measured_requests":   args.measured_requests,
            "total_requests":      total,
            "successful_requests": success,
            "random_seed":         args.random_seed,
            "vllm_bench_raw":      raw,
        },
    }


def main() -> None:
    args = parse_args()
    port = resolve_port(args)

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_name = "vllm_bench_raw.json"
        cmd = build_cmd(args, port, tmp_dir, raw_name)

        print("=" * 72)
        print(f"[{args.case_id or 'N/A'}] mode={args.mode} "
              f"isl={args.isl} osl={args.osl} concurrency={args.concurrency} port={port}")
        print("cmd:", " ".join(cmd))
        print("=" * 72)

        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"WARNING: vllm bench serve exited {proc.returncode}", file=sys.stderr)

        raw_path = Path(tmp_dir) / raw_name
        if not raw_path.exists():
            print(f"ERROR: result file not found: {raw_path}", file=sys.stderr)
            sys.exit(1)

        raw = json.loads(raw_path.read_text(encoding="utf-8"))

    summary = convert(raw, args, port)

    print("=" * 72)
    print("Result:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 72)

    if args.output_file:
        out = Path(args.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
