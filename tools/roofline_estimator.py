#!/usr/bin/env python3
"""Tiny Roofline-model calculator for interview/back-of-envelope reasoning.

Example:
  python tools/roofline_estimator.py \
    --flops 2e12 --bytes 1e11 --peak-tflops 100 --bandwidth-gbs 2000
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate a simple Roofline upper bound.")
    p.add_argument("--flops", type=float, required=True, help="Total FLOPs in the workload/kernel")
    p.add_argument("--bytes", type=float, required=True, help="Bytes moved from the modeled slow-memory level")
    p.add_argument("--peak-tflops", type=float, required=True, help="Peak compute throughput in TFLOP/s")
    p.add_argument("--bandwidth-gbs", type=float, required=True, help="Memory bandwidth in GB/s (decimal GB)")
    args = p.parse_args()

    if args.bytes <= 0 or args.flops < 0 or args.peak_tflops <= 0 or args.bandwidth_gbs <= 0:
        raise SystemExit("Inputs must be positive (FLOPs may be zero).")

    ai = args.flops / args.bytes
    peak_flops_s = args.peak_tflops * 1e12
    bandwidth_bytes_s = args.bandwidth_gbs * 1e9
    bandwidth_roof = bandwidth_bytes_s * ai
    attainable = min(peak_flops_s, bandwidth_roof)
    ridge = peak_flops_s / bandwidth_bytes_s
    bottleneck = "bandwidth" if bandwidth_roof < peak_flops_s else "compute"

    ideal_time_compute = args.flops / peak_flops_s
    ideal_time_memory = args.bytes / bandwidth_bytes_s
    ideal_time = max(ideal_time_compute, ideal_time_memory)

    print("Simple Roofline estimate")
    print("------------------------")
    print(f"arithmetic intensity : {ai:.3f} FLOP/byte")
    print(f"ridge point          : {ridge:.3f} FLOP/byte")
    print(f"bandwidth roof       : {bandwidth_roof / 1e12:.3f} TFLOP/s")
    print(f"compute roof         : {args.peak_tflops:.3f} TFLOP/s")
    print(f"attainable upper bound: {attainable / 1e12:.3f} TFLOP/s")
    print(f"first-order regime   : {bottleneck}-limited")
    print(f"idealized lower-bound time: {ideal_time * 1e3:.3f} ms")
    print("\nCaveat: real kernels may be limited by latency, occupancy, instruction mix,")
    print("cache behavior, synchronization, launch overhead, or imperfect utilization.")


if __name__ == "__main__":
    main()
