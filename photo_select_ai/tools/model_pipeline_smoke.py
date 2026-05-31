from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_pipeline.smoke_test import parse_thresholds, result_to_json, run_model_pipeline_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="PhotoSelect model_pipeline_v1 smoke test")
    parser.add_argument("input_dir", help="Photo directory")
    parser.add_argument("--backend", choices=["auto", "openclip", "fallback", "mock"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--similarity-threshold", type=float, default=0.86)
    parser.add_argument(
        "--grouping-strategy",
        choices=["connected_components", "complete_linkage", "average_linkage", "sequence_constrained"],
        default="complete_linkage",
    )
    parser.add_argument("--group-min-similarity", type=float, default=0.92)
    parser.add_argument("--max-group-size", type=int, default=25)
    parser.add_argument(
        "--threshold-sweep",
        default="",
        help="Comma-separated thresholds, e.g. 0.78,0.82,0.86,0.90,0.94",
    )
    parser.add_argument("--manual-groups-csv", default="", help="Optional CSV with columns: file_name,manual_group")
    parser.add_argument("--force-refresh-cache", action="store_true")
    parser.add_argument("--output-report", default="")
    parser.add_argument("--max-photos", type=int, default=0)
    args = parser.parse_args()

    result = run_model_pipeline_smoke(
        input_dir=Path(args.input_dir),
        backend=args.backend,
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold,
        force_refresh_cache=args.force_refresh_cache,
        output_report=Path(args.output_report) if args.output_report else None,
        max_photos=args.max_photos or None,
        thresholds=parse_thresholds(args.threshold_sweep) or None,
        manual_groups_csv=Path(args.manual_groups_csv) if args.manual_groups_csv else None,
        grouping_strategy=args.grouping_strategy,
        group_min_similarity_threshold=args.group_min_similarity,
        max_group_size=args.max_group_size,
    )
    print(result_to_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
