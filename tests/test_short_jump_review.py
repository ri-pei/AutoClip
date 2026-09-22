"""Synthetic boundary cases; no Sora filenames, frame numbers or pictures."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_refinement import refine_segments, save_review_sheet
from settings import load_settings
from short_jump_review import review_short_jumps
from time_mapping import segment_source_times, source_frame_samples


def fixture(root, length=3, flat=True, foreign=False, genuine=False, fps=60, duplicate=False):
    rng = np.random.default_rng(64017)
    source = rng.integers(0, 230, (250, 64, 64), dtype=np.uint8)
    remote = source.copy()
    if flat:
        source[30] = 255
        source[200] = 253
        remote[200] = 253
    if genuine:
        source[200] = 0 if flat else rng.integers(0, 256, (64, 64), dtype=np.uint8)
    if duplicate:
        source[200] = source[30]
    before = np.repeat(np.arange(20, 30), 2)
    after = np.repeat(np.arange(31, 41), 2)
    truth_middle = source[200] if genuine else source[30]
    query = np.concatenate([source[before], np.repeat(truth_middle[None], length, axis=0), source[after]])
    for name, frames in (("a", source), ("b", remote), ("q", query)):
        (root / name).mkdir()
        np.save(root / name / "compact_frames.npy", frames)
        (root / name / "compact_frames.json").write_text(json.dumps({"timestamps_ms": (np.arange(len(frames)) * 1000 / 30).tolist()}))
    edited = pd.DataFrame(dict(edited_frame_number=np.arange(len(query)), edited_timestamp_ms=np.arange(len(query))*1000/fps))
    segments = pd.DataFrame([
        dict(edited_start_frame=0, edited_end_frame=19, original_video_name="a", source_start_time_ms=20/30*1000,
             speed=fps/60, edited_start_time_ms=0, edited_end_time_ms=19*1000/fps),
        dict(edited_start_frame=20, edited_end_frame=19+length, original_video_name="b" if foreign else "a",
             source_start_time_ms=200/30*1000, speed=0, edited_start_time_ms=20*1000/fps, edited_end_time_ms=(19+length)*1000/fps),
        dict(edited_start_frame=20+length, edited_end_frame=39+length, original_video_name="a", source_start_time_ms=31/30*1000,
             speed=fps/60, edited_start_time_ms=(20+length)*1000/fps, edited_end_time_ms=(39+length)*1000/fps)])
    segments["edited_total_frames"] = len(query)
    return refine_segments(segments, edited, root, "q", sampling="floor")


class ShortJumpReviewTests(unittest.TestCase):
    def check_export(self, segments, details, root):
        target_times = details.edited_timestamp_ms.to_numpy(float)/1000
        count = 0
        for row in segments.to_dict("records"):
            start, end = int(row["edited_start_frame"]), int(row["edited_end_frame"]) + 1
            times = np.asarray(json.loads((root/row["original_video_name"]/"compact_frames.json").read_text())["timestamps_ms"])/1000
            mapped = segment_source_times(row, target_times)
            ids, _, weights = source_frame_samples(times, mapped, row["frame_sampling"])
            np.testing.assert_array_equal(ids, details.original_frame_number.iloc[start:end])
            np.testing.assert_allclose(weights, details.blend_weight.iloc[start:end], atol=1e-8)
            count += end - start
        self.assertEqual(count, len(details))

    def test_flat_excursion_is_bridged_and_neighbours_are_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_segments, old_details, old_report = fixture(root)
            original = old_details.copy(deep=True)
            segments, details, report, audit = review_short_jumps(old_segments, old_details, old_report, root, "q")
            self.assertEqual(audit["repaired_frames"], 3)
            self.assertEqual(len(segments), 1)
            self.assertEqual(details.original_frame_number.iloc[20:23].tolist(), [30]*3)
            self.assertEqual(details.status.iloc[20:23].tolist(), ["ambiguous"]*3)
            outside = np.r_[np.arange(20), np.arange(23, len(details))]
            for field in ("original_frame_number", "blend_weight", "pixel_mae"):
                np.testing.assert_array_equal(details[field].iloc[outside], original[field].iloc[outside])
            pd.testing.assert_frame_equal(old_details, original)
            self.assertNotIn("short_jump_review", old_report)
            self.assertEqual(report["assigned_frames"], old_report["assigned_frames"])
            self.assertEqual(sum(report["status_counts"].values()), len(details))
            json.dumps(audit, allow_nan=False)
            self.check_export(segments, details, root)

    def test_true_one_and_three_frame_cuts_are_kept(self):
        for length in (1, 3):
            for flat in (False, True):
                with self.subTest(length=length, flat=flat), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    old_segments, old_details, old_report = fixture(root, length=length, flat=flat, genuine=True)
                    segments, details, _, audit = review_short_jumps(old_segments, old_details, old_report, root, "q")
                    self.assertEqual(audit["repaired_frames"], 0)
                    self.assertEqual(len(segments), 3)
                    np.testing.assert_array_equal(details.original_frame_number, old_details.original_frame_number)
                    self.check_export(segments, details, root)

    def test_low_information_is_not_required_for_clear_visual_correction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_segments, old_details, old_report = fixture(root, flat=False)
            _, details, _, audit = review_short_jumps(old_segments, old_details, old_report, root, "q")
            self.assertEqual(audit["repaired_frames"], 3)
            self.assertFalse(details.low_information.iloc[20:23].any())

    def test_cross_source_white_collision_preserves_affine_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_segments, old_details, old_report = fixture(root, foreign=True)
            segments, details, _, audit = review_short_jumps(old_segments, old_details, old_report, root, "q")
            self.assertEqual(audit["repaired_frames"], 3)
            self.assertEqual(details.original_video_name.iloc[20:23].tolist(), ["a"]*3)
            self.assertEqual(details.affine_original_video_name.iloc[20:23].tolist(), ["b"]*3)
            self.assertTrue(details.frame_adjustment.iloc[20:23].isna().all())
            save_review_sheet(details, root, "q", root/"review.jpg", count=len(details))
            self.check_export(segments, details, root)

    def test_no_bridge_across_unknown_gap_or_unreliable_anchor(self):
        for kind in ("gap", "anchor", "other_source", "reverse"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                old_segments, old_details, old_report = fixture(root)
                if kind == "gap":
                    old_segments.loc[0, "edited_end_frame"] = 18
                    old_details.loc[19, "status"] = "unmatched"
                elif kind == "anchor":
                    old_details.loc[19, "status"] = "ambiguous"
                elif kind == "other_source":
                    old_segments.loc[2, "original_video_name"] = "b"
                else:
                    old_details.loc[23, "original_frame_number"] = 10
                _, _, _, audit = review_short_jumps(old_segments, old_details, old_report, root, "q")
                self.assertEqual(audit["repaired_frames"], 0)

    def test_duration_limit_uses_seconds_not_frame_count(self):
        for fps, length, expected in ((60, 3, 1), (24, 3, 0), (60, 6, 1), (60, 7, 0)):
            with self.subTest(fps=fps, length=length), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                old_segments, old_details, old_report = fixture(root, fps=fps, length=length)
                _, _, _, audit = review_short_jumps(old_segments, old_details, old_report, root, "q", max_duration_ms=100)
                self.assertEqual(audit["reviewed_clips"], expected)

    def test_merged_map_preserves_existing_blends_on_both_sides(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_segments, old_details, old_report = fixture(root)
            source = np.load(root/"a"/"compact_frames.npy")
            query = np.load(root/"q"/"compact_frames.npy")
            for i in (1, 3, 5, 24, 26, 28):
                frame = int(old_details.original_frame_number.iloc[i])
                mixed = source[frame].astype(float)*.75 + source[frame + 1]*.25
                query[i] = np.rint(mixed).astype(np.uint8)
                old_details.loc[i, "blend_weight"] = .25
                old_details.loc[i, "pixel_mae"] = float(np.abs(query[i]-mixed).mean())
            np.save(root/"q"/"compact_frames.npy", query)
            for row_id, row in old_segments.iterrows():
                start, end = int(row.edited_start_frame), int(row.edited_end_frame)+1
                chunk = old_details.iloc[start:end]
                values = (chunk.original_frame_number.to_numpy() + chunk.blend_weight.to_numpy()) / 30
                points = [[i, float(value*1000)] for i, value in enumerate(values)]
                points.append([len(values), points[-1][1]])
                old_segments.loc[row_id, "time_map"] = json.dumps(points)
                old_segments.loc[row_id, "frame_sampling"] = "frame-blending"
            segments, details, _, audit = review_short_jumps(old_segments, old_details, old_report, root, "q")
            self.assertEqual(audit["repaired_frames"], 3)
            self.assertEqual(segments.frame_sampling.tolist(), ["frame-blending"])
            outside = np.r_[np.arange(20), np.arange(23, len(details))]
            np.testing.assert_array_equal(details.blend_weight.iloc[outside], old_details.blend_weight.iloc[outside])
            self.check_export(segments, details, root)

    def test_near_equal_detailed_alternatives_are_flagged_not_rewritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_segments, old_details, old_report = fixture(root, flat=False, duplicate=True)
            # Equally good repeated detailed content does not justify rewriting.
            segments, details, _, audit = review_short_jumps(old_segments, old_details, old_report, root, "q")
            self.assertEqual(audit["repaired_frames"], 0)
            self.assertEqual(audit["unresolved_clips"], 1)
            self.assertEqual(len(segments), 3)
            self.assertEqual(details.status.iloc[20:23].tolist(), ["ambiguous"]*3)
            self.assertEqual(details.jump_review_action.iloc[20:23].tolist(), ["needs_review"]*3)

    def test_whole_frame_failure_cannot_hide_in_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_segments, old_details, old_report = fixture(root)
            # One real dark flash among two white frames: the majority alone is
            # insufficient, even if a user sets a permissive global MAE limit.
            path = root/"q"/"compact_frames.npy"
            query = np.load(path)
            query[21] = 0
            np.save(path, query)
            old_details.loc[20:22, "pixel_mae"] = 0.
            _, _, _, audit = review_short_jumps(old_segments, old_details, old_report, root, "q", review_mae=255)
            self.assertEqual(audit["repaired_frames"], 0)

    def test_invalid_settings_and_current_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder)/"config.toml"
            config.write_text('[paths]\nedited_video="mv.mp4"\n')
            self.assertTrue(load_settings(config)["short_jump_review"])
            for extra in ('frame_refinement=false', 'jump_max_duration_ms=0', 'jump_max_duration_ms=nan', 'short_jump_review="yes"'):
                config.write_text('[paths]\nedited_video="mv.mp4"\n[step4]\n' + extra + '\n')
                with self.assertRaises(ValueError):
                    load_settings(config)


if __name__ == "__main__":
    unittest.main()
