#!/usr/bin/env python3
"""Back-of-the-envelope KV-cache capacity calculator.

Example:
  python tools/kv_cache_calculator.py \
    --layers 32 --kv-heads 8 --head-dim 128 \
    --seq-len 16384 --bytes-per-element 2 --concurrency 16
"""

import argparse


def human_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        if abs(value) < 1024 or unit == units[-1]:
            return f"{value:.3f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate raw Transformer KV-cache memory.")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--kv-heads", type=int, required=True)
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--bytes-per-element", type=float, default=2.0,
                   help="2 for BF16/FP16, 1 for 8-bit, 0.5 for idealized 4-bit raw storage")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--kv-budget-gib", type=float, default=None,
                   help="Optional KV-only HBM budget; prints rough max concurrency")
    args = p.parse_args()

    per_token = 2 * args.layers * args.kv_heads * args.head_dim * args.bytes_per_element
    per_request = per_token * args.seq_len
    total = per_request * args.concurrency

    print("Raw KV-cache estimate")
    print("---------------------")
    print(f"per token / request : {human_bytes(per_token)}")
    print(f"per request         : {human_bytes(per_request)}")
    print(f"{args.concurrency} active requests : {human_bytes(total)}")

    if args.kv_budget_gib is not None:
        budget = args.kv_budget_gib * 1024**3
        max_req = int(budget // per_request) if per_request else 0
        print(f"rough requests in {args.kv_budget_gib:g} GiB KV budget: {max_req}")

    print("\nFormula:")
    print("KV bytes = 2 × layers × sequence × KV_heads × head_dim × bytes/element")
    print("\nNote: this is raw tensor storage. Runtime block granularity, metadata, workspace,")
    print("allocator reserve, graph buffers, and other model/runtime memory are not included.")


if __name__ == "__main__":
    main()
