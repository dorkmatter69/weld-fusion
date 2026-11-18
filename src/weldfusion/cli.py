"""Command line: scan a dataset folder, convert it to MCAP, inspect a result."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from .dataset import discover_samples, load_sensor_table
from .mcap_io import summarize_mcap, write_weld_mcap


def cmd_scan(args) -> int:
    samples = discover_samples(args.root)
    print(f"{len(samples)} welds under {args.root}")
    by_class = Counter(s.class_name for s in samples)
    for name, n in sorted(by_class.items()):
        print(f"  {name:36s} {n}")
    by_mod = Counter(",".join(s.modalities()) for s in samples)
    print("modalities present:")
    for combo, n in sorted(by_mod.items()):
        print(f"  {combo or '(none)':36s} {n}")
    first = next((s for s in samples if s.sensors), None)
    if first:
        table = load_sensor_table(first.sensors)
        print(f"sensor columns ({first.sensors.name}): {table.columns}")
        if table.dropped_columns:
            print(f"dropped non-signal columns: {table.dropped_columns}")
    return 0


def cmd_convert(args) -> int:
    samples = discover_samples(args.root)
    duplicate_stems = Counter(s.stem for s in samples)
    if args.limit:
        samples = samples[: args.limit]
    out_dir = Path(args.out)
    manifest, failures = [], 0
    for i, sample in enumerate(samples, 1):
        target = out_dir / f"{sample.output_name(duplicate_stems[sample.stem] > 1)}.mcap"
        try:
            alignment = write_weld_mcap(
                sample,
                target,
                sensor_rate_hz=args.sensor_hz,
                embed_every=args.embed_every,
                allow_unregistered_clocks=args.allow_unregistered_clocks,
            )
        except Exception as exc:  # keep going; report at the end
            failures += 1
            print(f"[{i}/{len(samples)}] FAILED {sample.stem}: {exc}", file=sys.stderr)
            continue
        manifest.append(
            {
                "stem": sample.stem,
                "source_subdir": sample.extra.get("relative_subdir"),
                "manifest_record": sample.extra.get("manifest"),
                "class_id": sample.class_id,
                "class_name": sample.class_name,
                "mcap": target.name,
                "modalities": sample.modalities(),
                "alignment": alignment,
            }
        )
        offset = alignment["start_offset_sensor_minus_video_s"]
        note = f"offset {offset:+.2f}s" if offset is not None else "offset n/a"
        print(f"[{i}/{len(samples)}] {sample.stem}  {alignment['sensor_rate_hz']:.2f} Hz  {note}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} MCAP files and manifest.json to {out_dir}; {failures} failed")
    return 1 if failures else 0


def cmd_inspect(args) -> int:
    print(json.dumps(summarize_mcap(args.file), indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="weldfusion")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="count welds, classes and modalities")
    p.add_argument("root")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("convert", help="write one MCAP per weld")
    p.add_argument("root")
    p.add_argument("out")
    p.add_argument("--sensor-hz", type=float, default=None,
                   help="sensor rate; default spreads rows over the video duration")
    p.add_argument("--embed-every", type=int, default=0,
                   help="embed every Nth frame as JPEG for viewing (0 = none)")
    p.add_argument("--limit", type=int, default=0, help="convert only the first N welds")
    p.add_argument("--allow-unregistered-clocks", action="store_true",
                   help="exploratory only: put separate sensor and video time origins on one MCAP axis")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("inspect", help="summarise one MCAP file")
    p.add_argument("file")
    p.set_defaults(func=cmd_inspect)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
