"""Frame selection, export and rejection tests independent of the Sora edit."""

from fractions import Fraction
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frame_refinement import refine_frame_path, refine_sampling_path, refine_segments
from compact_frames import extract_compact
from render_reconstruction import comparison_filter, render, render_timestamps
from settings import load_settings
from step4 import load_all_original_frames_data
from step5 import csv_to_fcpxml, get_fcpxml_time_params
from time_mapping import compress_frame_map, parse_time_map, sample_source_frames, segment_source_times


class FrameRefinementTests(unittest.TestCase):
    def test_compression_preserves_frames_with_freezes_and_variable_timestamps(self):
        rng = np.random.default_rng(8452)
        for count in (1, 2, 71, 541):
            times = np.r_[0, np.cumsum(rng.uniform(.025, .044, 2000))]
            ids = 23 + np.cumsum(rng.choice([0, 0, 1, 1, 2, 3], count))
            preferred = times[ids] + rng.uniform(-.01, .05, count)
            points = compress_frame_map(ids, times, preferred)
            parsed = parse_time_map(json.dumps(points), count)
            actual = sample_source_frames(times, np.interp(np.arange(count), parsed[:, 0], parsed[:, 1]) / 1000)
            np.testing.assert_array_equal(actual, ids)
            self.assertLessEqual(len(points), count + 1)

    def test_local_path_recovers_duplicate_drop_cadence_without_new_cuts(self):
        rng = np.random.default_rng(58)
        source = rng.integers(0, 256, (180, 64, 64), dtype=np.uint8)
        source_times = np.arange(len(source)) / 30
        prediction = 20 / 30 + np.arange(160) / 60
        baseline = sample_source_frames(source_times, prediction)
        truth = baseline.copy()
        truth[35:60] += 1
        truth[112:136] -= 1
        # These changes are monotone; random texture makes identity unambiguous.
        query = source[truth]
        result, error, original_error, _ = refine_frame_path(query, source, source_times, prediction)
        np.testing.assert_array_equal(result, truth)
        self.assertEqual(float(error.max()), 0.)
        self.assertGreater(float(original_error.max()), 50.)

    def test_identical_frames_retain_prior_instead_of_inventing_motion(self):
        picture = np.random.default_rng(14).integers(0, 256, (1, 64, 64), dtype=np.uint8)
        source = np.repeat(picture, 100, axis=0)
        times = np.arange(100) / 24
        predicted = .5 + np.arange(40) / 30
        actual, _, _, _ = refine_frame_path(source[:40], source, times, predicted)
        np.testing.assert_array_equal(actual, sample_source_frames(times, predicted))

    def test_auto_sampling_recovers_mixed_frames_but_keeps_sharp_frames(self):
        rng = np.random.default_rng(380)
        source = rng.integers(0, 256, (90, 64, 64), dtype=np.uint8)
        times = np.arange(90) / 30
        positions = 20 + np.arange(48) / 2
        left = np.floor(positions).astype(int)
        weights = positions - left
        query = np.rint(source[left] * (1-weights[:, None, None])
                        + source[left+1] * weights[:, None, None]).astype(np.uint8)
        ids, blend, error, _, _, sampling = refine_sampling_path(query, source, times, positions/30)
        self.assertEqual(sampling, 'frame-blending')
        np.testing.assert_allclose(ids + blend, positions, atol=.01)
        self.assertLess(float(error.max()), .6)
        sharp = np.clip(source[left].astype(int)+2, 0, 255).astype(np.uint8)
        ids, blend, _, _, _, sampling = refine_sampling_path(sharp, source, times, positions/30)
        self.assertEqual(sampling, 'floor')
        np.testing.assert_array_equal(ids, left)
        self.assertFalse(blend.any())

    def test_invalid_maps_and_out_of_source_times_fail(self):
        for value in ([[0, 0], [4, -1]], [[1, 0], [4, 1]], [[0, 0], [3, 1]],
                      [[0, 0], [2, 2], [1, 3], [4, 4]]):
            with self.assertRaises(ValueError):
                parse_time_map(value, 4)
        for target in (-.1, 1., float('nan')):
            with self.assertRaises(ValueError):
                sample_source_frames(np.arange(24) / 24, [target])

    def test_local_blend_evidence_is_not_diluted_by_a_long_sharp_tail(self):
        rng = np.random.default_rng(7583)
        source = rng.integers(0, 256, (350, 64, 64), dtype=np.uint8)
        times = np.arange(len(source)) / 30
        positions = 20 + np.arange(600) / 2
        ids = np.floor(positions).astype(int)
        query = source[ids].copy()
        mixed = np.arange(11, 49, 2)  # Fewer than 5% of the entire clip.
        query[mixed] = np.rint((source[ids[mixed]].astype(float) + source[ids[mixed]+1]) / 2).astype(np.uint8)
        actual, weights, error, _, _, sampling = refine_sampling_path(query, source, times, positions / 30)
        self.assertEqual(sampling, 'frame-blending')
        np.testing.assert_array_equal(actual, ids)
        np.testing.assert_allclose(weights[mixed], .5, atol=.01)
        self.assertLess(float(error.max()), .6)
        sharp = np.ones(len(query), dtype=bool)
        sharp[mixed] = False
        self.assertFalse(weights[sharp].any())

    def test_unrelated_local_pixels_are_reported_for_review(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(98)
            for name in ('s', 'q'):
                (root / name).mkdir()
                count = 30 if name == 's' else 20
                np.save(root / name / 'compact_frames.npy', rng.integers(0, 256, (count, 64, 64), dtype=np.uint8))
                (root / name / 'compact_frames.json').write_text(json.dumps({'timestamps_ms': (np.arange(count)*40).tolist()}))
            edited = pd.DataFrame(dict(edited_frame_number=range(20), edited_timestamp_ms=np.arange(20)*40))
            rows = pd.DataFrame([dict(edited_start_frame=0, edited_end_frame=19, original_video_name='s',
                                      source_start_time_ms=80, speed=1)])
            _, diagnostics, report = refine_segments(rows, edited, root, 'q')
            self.assertEqual(diagnostics.status.tolist(), ['review']*20)
            self.assertEqual(report['assigned_fraction'], 1.)
            self.assertGreater(report['pixel_error_after']['mean'], 70.)

    def test_source_scope_and_stem_substring(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('mv', 'mv_source', 'stale'):
                (root / name).mkdir()
                pd.DataFrame([dict(video_name=name, frame_number=0, timestamp_ms=0,
                                   phash='0'*64)]).to_csv(root / name / f'{name}_phash.csv', index=False)
            backup = root / 'backup' / 'mv_source'
            backup.mkdir(parents=True)
            (backup / 'mv_source_phash.csv').write_bytes((root / 'mv_source' / 'mv_source_phash.csv').read_bytes())
            frames = load_all_original_frames_data(root, 'mv', {'mv_source'})
            self.assertEqual(frames.original_video_name.tolist(), ['mv_source'])
            with self.assertRaises(FileNotFoundError):
                load_all_original_frames_data(root, 'mv', {'missing'})

    def test_fps_does_not_confuse_near_sixty_with_ntsc(self):
        for rate, expected in ((59.9995, '1/60s'), (60, '1/60s'), (59.94, '1001/60000s'),
                               (23.976, '1001/24000s'), (24, '1/24s'), (29.999, '1/30s')):
            self.assertEqual(get_fcpxml_time_params(rate)[0], expected)

    def test_renderer_reuses_timestamps_only_for_the_same_unchanged_media(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / 'source.webm'
            video.write_bytes(b'fixture')
            (root / video.stem).mkdir()
            stat = video.stat()
            manifest = dict(signature=dict(version=2, path=str(video.resolve()), size=stat.st_size,
                                           mtime_ns=stat.st_mtime_ns), timestamps_ms=[0, 33, 67])
            (root / video.stem / 'compact_frames.json').write_text(json.dumps(manifest))
            settings = dict(output_dir=str(root), frame_storage='compact')
            with patch('render_reconstruction.get_frame_timestamps_map_json') as probe:
                np.testing.assert_allclose(render_timestamps(settings, video), [0, .033, .067])
                probe.assert_not_called()
                video.write_bytes(b'different fixture')
                with self.assertRaises(ValueError):
                    render_timestamps(settings, video)
                probe.assert_not_called()

    def test_render_temporary_file_cannot_overwrite_a_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'sources').mkdir()
            source = root / 'sources' / 'episode.partial.mp4'
            source.write_bytes(b'input media')
            edited = root / 'mv.mp4'
            edited.touch()
            settings = dict(source_dir=str(root / 'sources'), edited_video_path=str(edited))
            with self.assertRaisesRegex(ValueError, 'must not overwrite'):
                render(settings, root / 'sources' / 'episode.mp4')
            self.assertEqual(source.read_bytes(), b'input media')

    def test_compact_cache_invalidates_when_ffmpeg_backend_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / 'source.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=red:s=32x32:r=24',
                            '-frames:v', '4', '-threads', '1', '-c:v', 'libx264', str(video)], check=True)
            destination = root / 'cache'
            extract_compact(video, destination, [])
            original = json.loads((destination / 'compact_frames.json').read_text())
            self.assertEqual(original['signature']['version'], 2)
            with patch('compact_frames.get_frame_timestamps_map_json', side_effect=AssertionError('unnecessary decode')):
                extract_compact(video, destination, [])
            different = dict(original['signature']['ffmpeg'], build_sha256='changed-for-test')
            with patch('compact_frames.get_ffmpeg_identity', return_value=different):
                extract_compact(video, destination, [])
            rebuilt = json.loads((destination / 'compact_frames.json').read_text())
            self.assertEqual(rebuilt['signature']['ffmpeg'], different)

    def test_comparison_pairs_frame_numbers_with_different_container_clocks(self):
        # Two equal image sequences have different rounded timestamps. The
        # comparison must not display one side a frame early or late.
        inputs = ["color=black:s=32x32:r=60:d=1,format=gray,geq=lum='N*3',"
                  f"settb=expr=1/{clock},setpts=N/(60*TB)" for clock in (16000, 15360)]
        command = ['ffmpeg', '-v', 'error', '-filter_complex_threads', '1']
        for value in inputs:
            command += ['-f', 'lavfi', '-i', value]
        command += ['-filter_complex', comparison_filter(1, 60), '-map', '[v]', '-r', '60',
                    '-threads', '1', '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1']
        frames = np.frombuffer(subprocess.check_output(command), dtype=np.uint8).reshape(-1, 64, 32)
        self.assertEqual(len(frames), 60)
        np.testing.assert_array_equal(frames[:, :32], frames[:, 32:])
        np.testing.assert_array_equal(frames[:, 0, 0], np.arange(60)*3)

    def test_export_and_renderer_sample_identical_map_and_music_has_no_retimed_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'sources').mkdir()
            source = root/'sources'/'a # ü.webm'
            edited = root/'mv.mp4'
            source.touch(); edited.touch()
            config = root/'config.toml'
            config.write_text('[paths]\nedited_video="mv.mp4"\nsource_dir="sources"\noutput_dir="."\n'
                              '[step5]\nframe_rate=60\nproject_name="test"\n')
            settings = load_settings(config)
            times = np.arange(100) / 24
            ids = np.r_[np.arange(12)//2+10, [15]*6]
            points = compress_frame_map(ids, times, times[ids]+.01)
            row = dict(edited_start_frame=0, edited_end_frame=len(ids)-1, edited_total_frames=len(ids),
                       original_video_name=source.stem, original_start_frame=int(ids[0]), speed=.8,
                       source_start_time_ms=points[0][1], source_end_time_ms=points[-1][1],
                       time_map=json.dumps(points))
            pd.DataFrame([row]).to_csv(root/settings['final_segments_csv'], index=False)
            metadata = dict(width=640, height=360, avg_frame_rate=60)
            audio = dict(sample_rate=48000, channels=2, duration_seconds=1)
            with patch('step5.get_video_metadata', return_value=metadata), patch('step5.get_audio_metadata', return_value=audio):
                csv_to_fcpxml(settings)
            xml = ET.parse(root/'test.fcpxml').getroot()
            pts = xml.findall('.//timeMap/timept')
            seconds = lambda value: float(Fraction(value.removesuffix('s')))
            xx = np.array([seconds(point.get('time')) for point in pts])
            yy = np.array([seconds(point.get('value')) for point in pts])
            exported = sample_source_frames(times, np.interp(np.arange(len(ids))/60, xx, yy))
            rendered = sample_source_frames(times, segment_source_times(row, np.arange(len(ids))/60))
            np.testing.assert_array_equal(exported, ids)
            np.testing.assert_array_equal(rendered, ids)
            parent = {child: element for element in xml.iter() for child in element}
            audio_clip = xml.find('.//asset-clip[@srcEnable="audio"]')
            while audio_clip in parent:
                audio_clip = parent[audio_clip]
                self.assertIsNone(audio_clip.find('timeMap'))
            self.assertIn('%23', xml.find('./resources/asset/media-rep').get('src'))


if __name__ == '__main__':
    unittest.main()
