"""Checks for current experiment provenance, coverage and scoring (no old baseline)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fixture_support import (check_fixture, finish_fixture, fixture_identity,
                             frame_metrics, predict_frames)
import benchmark_retime


class ExperimentTests(unittest.TestCase):
    def test_fixture_cache_requires_identity_and_complete_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'fixture'
            identity = {'recipe': 1}
            self.assertFalse(check_fixture(root, identity, ['video.mp4']))
            (root/'video.mp4').write_bytes(b'fixture')
            finish_fixture(root, identity)
            self.assertTrue(check_fixture(root, identity, ['video.mp4']))
            with self.assertRaises(ValueError):
                check_fixture(root, {'recipe': 2}, ['video.mp4'])
            with self.assertRaises(ValueError):
                check_fixture(root, identity, ['missing.toml'])
            self.assertFalse(check_fixture(root, identity, ['video.mp4'], force=True))
            self.assertFalse((root/'fixture.json').exists())

    def test_input_content_and_toolchain_are_part_of_fixture_identity(self):
        with tempfile.TemporaryDirectory() as folder, patch('fixture_support.get_ffmpeg_identity', return_value={'version': 'a'}):
            media = Path(folder)/'input.mp4'
            media.write_bytes(b'first')
            before = fixture_identity([media], {'seed': 1})
            media.write_bytes(b'other')
            after = fixture_identity([media], {'seed': 1})
            self.assertNotEqual(before, after)
            with patch('fixture_support.get_ffmpeg_identity', return_value={'version': 'b'}):
                self.assertNotEqual(after, fixture_identity([media], {'seed': 1}))

    def test_scoring_counts_wrong_source_blends_and_unknown_matches(self):
        result = frame_metrics(['a', 'a', 'a', ''], [1, 2, 3, -1],
                               np.array(['b', 'a', 'a', 'a']), np.array([1, 2, 3, 4]),
                               np.array([0, .5, 0, 0]))
        self.assertAlmostEqual(result['exact_frame_accuracy'], 1/3)
        self.assertEqual(result['false_matches'], 1)
        self.assertEqual(result['blended_frames'], 1)

    def test_prediction_preserves_blending_and_gaps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'a').mkdir()
            (root/'a'/'compact_frames.json').write_text(json.dumps({'timestamps_ms': [0, 40, 80, 120]}))
            rows = [dict(edited_start_frame=1, edited_end_frame=2, original_video_name='a',
                         speed=1, source_start_time_ms=20, time_map='[[0,20],[1,40],[2,60]]',
                         frame_sampling='frame-blending')]
            path = root/'segments.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            _, names, ids, weights = predict_frames(path, root, np.arange(4)/25)
            self.assertEqual(names.tolist(), ['', 'a', 'a', ''])
            self.assertEqual(ids.tolist(), [-1, 0, 1, -1])
            np.testing.assert_allclose(weights, [0, .5, 0, 0])

    def test_benchmark_revalidates_caches_and_runs_refined_and_reviewed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'development.toml').write_text('[paths]\nedited_video="development.mp4"\n')
            output = root/'benchmark_development'
            output.mkdir()
            (output/'preparation.json').write_text('{"obsolete_timing": 999}')
            modes = []
            with patch.object(benchmark_retime.step1, 'main_step1') as first, \
                 patch.object(benchmark_retime.step2, 'main_step2') as second, \
                 patch.object(benchmark_retime.step3, 'main_step3'), \
                 patch.object(benchmark_retime.step4, 'main_step4', side_effect=lambda s: modes.append((s['frame_refinement'], s['short_jump_review']))), \
                 patch.object(benchmark_retime.step5, 'csv_to_fcpxml') as export, \
                 patch.object(benchmark_retime, 'get_ffmpeg_identity', return_value={}), \
                 patch.object(benchmark_retime, 'evaluate', return_value={}):
                result = benchmark_retime.run('development', root, variants=('refined', 'reviewed'), retrieval_benchmark=False)
            first.assert_called_once()
            second.assert_called_once()
            self.assertEqual(export.call_count, 2)
            self.assertEqual(modes, [(True, False), (True, True)])
            self.assertNotIn('obsolete_timing', result['preparation_seconds'])


if __name__ == '__main__':
    unittest.main()
