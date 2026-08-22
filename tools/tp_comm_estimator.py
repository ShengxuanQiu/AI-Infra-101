#!/usr/bin/env python3
"""Simple tensor-parallel communication-time estimator.

This is intentionally a first-order model:
  time_per_collective ≈ latency + message_bytes / effective_bandwidth

Example:
  python tools/tp_comm_estimator.py \
    --message-mib 16 --bandwidth-gbs 400 --latency-us 5 \
    --collectives-per-layer 2 --layers 80
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate first-order TP communication overhead.")
    p.add_argument("--message-mib", type=float, required=True,
                   help="Effective bytes transferred per collective in MiB for this simplified model")
    p.add_argument("--bandwidth-gbs", type=float, required=True,
                   help="Effective link/collective bandwidth in GB/s")
    p.add_argument("--latency-us", type=float, default=0.0,
                   help="Effective startup latency per collective in microseconds")
    p.add_argument("--collectives-per-layer", type=float, default=2.0)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--overlap", type=float, default=0.0,
                   help="Idealized hidden fraction of communication [0,1]")
    args = p.parse_args()

    if not 0.0 <= args.overlap <= 1.0:
        raise SystemExit("--overlap must be in [0,1]")

    msg_bytes = args.message_mib * 1024**2
    bw_bytes_s = args.bandwidth_gbs * 1e9
    latency_s = args.latency_us * 1e-6

    per_collective = latency_s + msg_bytes / bw_bytes_s
    n_collectives = args.collectives_per_layer * args.layers
    raw_total = per_collective * n_collectives
    exposed_total = raw_total * (1.0 - args.overlap)

    print("First-order TP communication estimate")
    print("-------------------------------------")
    print(f"message size          : {args.message_mib:.3f} MiB")
    print(f"per-collective time   : {per_collective * 1e6:.3f} us")
    print(f"collectives/forward   : {n_collectives:g}")
    print(f"raw communication     : {raw_total * 1e3:.3f} ms")
    print(f"assumed hidden overlap: {args.overlap * 100:.1f}%")
    print(f"exposed communication : {exposed_total * 1e3:.3f} ms")
    print("\nCaveat: real AllReduce/AllGather/ReduceScatter cost depends on TP degree,")
    print("collective algorithm, topology, contention, message size, synchronization,")
    print("and whether independent compute actually exists to overlap communication.")


if __name__ == "__main__":
    main()
