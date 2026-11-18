"""Export the preserved MuJoCo + MockPLC run into the browser WeldEpisode contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from weld_twin.episode import run_episode
from weld_twin.plc import MockPLC


def export_simulation(seed: int = 0, fault: str = "none") -> dict:
    if fault not in ("none", "stale_vision", "outlier"):
        raise ValueError("supported MuJoCo faults: none, stale_vision, outlier")
    trace = []
    args = {"stall_after_s": 5.0} if fault == "stale_vision" else (
        {"outlier_at_s": 5.0} if fault == "outlier" else {}
    )
    result = run_episode(seed, MockPLC(), trace=trace.append, **args)
    return {
        "schema_version": "1.0.0", "episode_id": f"mujoco-{seed}-{fault}",
        "source": {"kind": "simulation", "generator": "MuJoCo WeldSim + Python MockPLC"},
        "process": {"weld_type": "flat_butt_coupon", "position": "simulation_only"},
        "recipe": {},
        "timeline": {"time_basis": "simulation_elapsed_s", "clock_alignment_verified": True},
        "streams": {}, "telemetry": trace,
        "events": [{"time_s": trace[-1]["time_s"], "type": "simulated_fault", "cause": fault}]
                  if result.final_state == 3 else [],
        "reviews": [],
        "metrics": {"seam_rms_mm": result.rms_mm, "seam_max_mm": result.max_mm},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fault", choices=("none", "stale_vision", "outlier"), default="none")
    args = parser.parse_args(argv)
    episode = export_simulation(args.seed, args.fault)
    args.output.write_text(json.dumps(episode, indent=2))
    print(f"Wrote {len(episode['telemetry'])} MuJoCo/MockPLC steps to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
