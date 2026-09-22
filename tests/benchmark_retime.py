"""Benchmark the current pipeline and explicit ablations using independent truth."""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from settings import load_settings
from compact_frames import get_ffmpeg_identity
from fast_matching import pack_hashes, search_hashes
from fixture_support import predict_frames, frame_metrics
import step1
import step2
import step3
import step4
import step5

WORK = ROOT / "tests" / ".work" / "retime"
VARIANTS = {
    "affine": ("affine", False, False),
    "affine_ransac": ("affine_ransac", False, False),
    "affine_pixels": ("affine_pixels", False, False),
    "refined": ("affine_pixels", True, False),
    "reviewed": ("affine_pixels", True, True),
}


def evaluate(csv_path, truth_path):
    truth = json.loads(Path(truth_path).read_text())
    output = Path(csv_path).parent
    edited_name = Path(truth_path).name.removesuffix(".truth.json")
    edited_times = np.asarray(json.loads((output/edited_name/"compact_frames.json").read_text())["timestamps_ms"]) / 1000
    if len(edited_times) != len(truth["source_frames"]):
        raise ValueError("Target cache and truth have different frame counts")
    rows, names, ids, weights = predict_frames(csv_path, output, edited_times)
    metrics = frame_metrics(["reference"] * len(ids), truth["source_frames"], names, ids, weights)
    cuts = rows.edited_start_frame.to_numpy(int)
    expected_cuts = np.array([s["edited_start_frame"] for s in truth["segments"]])
    metrics.update(output_clips=len(rows), truth_clips=len(expected_cuts),
                   cut_max_error=int(np.abs(cuts[:, None] - expected_cuts).min(axis=0).max()) if len(cuts) else None,
                   cuts_exact=bool(np.array_equal(cuts, expected_cuts)))
    return metrics


def compare_retrieval(database, queries):
    results = []
    # NumPy is an independent exact Hamming reference, not an old pipeline.
    reference, _ = search_hashes(database, queries, 20, "numpy")
    for backend in ("numpy", "balltree_batch", "faiss", "faiss_hnsw"):
        try:
            timings = []
            for _ in range(3):
                start = time.perf_counter()
                distances, _ = search_hashes(database, queries, 20, backend)
                timings.append(time.perf_counter() - start)
            results.append(dict(backend=backend, runs_seconds=timings,
                                distance_agreement=float(np.mean(distances == reference))))
        except (RuntimeError, ImportError) as error:
            results.append(dict(backend=backend, skipped=str(error)))
    return results


def run(case, work=WORK, matcher="numpy", top_ks=(20,), variants=tuple(VARIANTS),
        retrieval_benchmark=True):
    work = Path(work)
    settings = load_settings(work / f"{case}.toml")
    output = work / f"benchmark_{case}"
    settings["output_dir"] = str(output)
    result = dict(case=case, platform=platform.platform(), python=sys.version,
                  ffmpeg=get_ffmpeg_identity(), matcher=matcher,
                  preparation_seconds={}, retrieval=[], variants=[],
                  timing_note="Every run validates Step 1/2 caches; preparation timings are for this run, not assumed cold timings.")
    # Always call the current cache validators, even if previous timings exist.
    for module in (step1, step2):
        start = time.perf_counter()
        getattr(module, "main_" + module.__name__)(settings)
        result["preparation_seconds"][module.__name__] = time.perf_counter() - start
    if retrieval_benchmark:
        query = pd.read_csv(output / case / f"{case}_phash.csv", dtype={"phash": str})
        source = pd.read_csv(output / "reference" / "reference_phash.csv", dtype={"phash": str})
        result["retrieval"] = compare_retrieval(pack_hashes(source.phash), pack_hashes(query.phash))
    settings["matcher"] = matcher
    report = work / f"{case}.benchmark.json"
    for k in top_ks:
        settings["top_k"] = k
        settings["coarse_match_csv"] = f"{matcher}_k{k}.coarse.csv"
        start = time.perf_counter()
        step3.main_step3(settings)
        matching_seconds = time.perf_counter() - start
        for variant in variants:
            alignment, refine, review = VARIANTS[variant]
            settings.update(alignment=alignment, frame_refinement=refine, short_jump_review=review)
            name = f"{matcher}_k{k}_{variant}"
            settings.update(final_segments_csv=name + ".segments.csv", project_name=name)
            start = time.perf_counter()
            step4.main_step4(settings)
            alignment_seconds = time.perf_counter() - start
            start = time.perf_counter()
            step5.csv_to_fcpxml(settings)
            export_seconds = time.perf_counter() - start
            metrics = evaluate(output / settings["final_segments_csv"], work / f"{case}.truth.json")
            result["variants"].append(dict(name=name, alignment=alignment, frame_refinement=refine,
                                           short_jump_review=review, top_k=k, matching_seconds=matching_seconds,
                                           alignment_seconds=alignment_seconds, export_seconds=export_seconds,
                                           **metrics))
            report.write_text(json.dumps(result, indent=2) + "\n")
            print(f"{name}: {metrics}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("development", "holdout", "validation"), default="development")
    parser.add_argument("--work-dir", type=Path, default=WORK)
    parser.add_argument("--matcher", choices=("numpy", "balltree_batch", "faiss", "faiss_hnsw"), default="numpy")
    parser.add_argument("--top-k", type=int, nargs="+", default=[20])
    parser.add_argument("--variants", choices=tuple(VARIANTS), nargs="+", default=list(VARIANTS))
    parser.add_argument("--skip-retrieval-benchmark", action="store_true")
    args = parser.parse_args()
    if any(not 1 <= k <= 100 for k in args.top_k):
        parser.error("--top-k must be between 1 and 100")
    run(args.case, args.work_dir, args.matcher, args.top_k, args.variants, not args.skip_retrieval_benchmark)
