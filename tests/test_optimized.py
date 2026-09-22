"""Focused correctness checks for optimized indexing and temporal rejection."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast_matching import pack_hashes, search_hashes
from compact_frames import hash_compact
from temporal_alignment import align_segments
from settings import load_settings


class OptimizedTests(unittest.TestCase):
    def test_packed_search_preserves_all_bits_and_exact_distances(self):
        rng=np.random.default_rng(19)
        source=rng.integers(0,256,(41,32),dtype=np.uint8)
        source[4]=source[3]
        query=np.vstack([source[3],np.zeros(32,dtype=np.uint8),source[10]])
        hashes=[row.tobytes().hex() for row in source]
        np.testing.assert_array_equal(pack_hashes(hashes),source)
        expected=np.unpackbits(source[None,:,:]^query[:,None,:],axis=2).sum(axis=2)
        for backend in ('numpy','balltree_batch','faiss'):
            try:
                d,i=search_hashes(source,query,5,backend,batch_size=2)
            except RuntimeError:
                if backend=='faiss':continue
                raise
            np.testing.assert_array_equal(d,np.sort(expected,axis=1)[:,:5])
            np.testing.assert_array_equal(d,np.take_along_axis(expected,i,axis=1))
        d,i=search_hashes(source,query,100,'numpy')
        self.assertEqual(i.shape,(3,41))

    def test_compact_dct_equals_imagehash_on_identical_pixels(self):
        frames=np.random.default_rng(5).integers(0,256,(7,64,64),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'source';root.mkdir()
            np.save(root/'compact_frames.npy',frames)
            (root/'compact_frames.json').write_text(json.dumps({'timestamps_ms':list(range(7))}))
            hash_compact('source',directory)
            rows=pd.read_csv(root/'source_phash.csv',dtype={'phash':str})
            self.assertEqual(set(rows.columns), {'video_name', 'frame_number', 'timestamp_ms', 'phash'})
            expected=[str(imagehash.phash(Image.fromarray(frame),hash_size=16)) for frame in frames]
            self.assertEqual(rows.phash.tolist(),expected)

    def test_no_candidates_produce_no_forced_match(self):
        edited=pd.DataFrame(dict(edited_frame_number=range(12),edited_timestamp_ms=np.arange(12)*40,
                                 edited_phash=['0'*64]*12,top_n_matches=[[] for _ in range(12)]))
        originals=pd.DataFrame(dict(original_video_name=['s']*12,original_frame_number=range(12),
                                    original_timestamp_ms=np.arange(12)*40,original_phash=['f'*64]*12))
        self.assertTrue(align_segments(edited,originals).empty)

    def test_multiple_sources_and_different_frame_rates(self):
        rng=np.random.default_rng(77)
        originals=[]
        hashes={}
        for name, fps in (("a",30),("b",24)):
            hashes[name]=[row.tobytes().hex() for row in rng.integers(0,256,(200,32),dtype=np.uint8)]
            originals.extend(dict(original_video_name=name,original_frame_number=i,
                                  original_timestamp_ms=i*1000/fps,original_phash=h)
                             for i,h in enumerate(hashes[name]))
        edited=[]
        truth=[]
        for name, fps, speed, first in (("b",24,.5,30),("a",30,1.5,50)):
            for local in range(36):
                index=first+int(np.floor(local*speed*fps/24+1e-8))
                truth.append((name,index))
                edited.append(dict(edited_frame_number=len(edited),edited_timestamp_ms=len(edited)*1000/24,
                                   edited_phash=hashes[name][index],top_n_matches=[dict(
                                       original_video_name=name,original_timestamp_ms=index*1000/fps,
                                       original_frame_number=index,phash_distance=0)]))
        result=align_segments(pd.DataFrame(edited),pd.DataFrame(originals))
        self.assertEqual(result.edited_start_frame.tolist(),[0,36])
        for row in result.itertuples():
            fps=30 if row.original_video_name=='a' else 24
            ids=np.floor(row.source_start_time_ms*fps/1000+row.speed*fps/24*np.arange(36)+1e-6).astype(int)
            expected=truth[row.edited_start_frame:row.edited_end_frame+1]
            self.assertEqual([(row.original_video_name,int(i)) for i in ids],expected)

    def test_piecewise_speed_change_is_not_discarded_by_premature_merge(self):
        rng=np.random.default_rng(721)
        source_fps,edited_fps,count=30,60,3000
        hashes=[row.tobytes().hex() for row in rng.integers(0,256,(count,32),dtype=np.uint8)]
        originals=pd.DataFrame(dict(original_video_name=['s']*count,
                                    original_frame_number=range(count),
                                    original_timestamp_ms=np.arange(count)*1000/source_fps,
                                    original_phash=hashes))
        edited=[]
        first_length,total=448,1539
        for frame in range(total):
            source_time=(30 + .823529*min(frame,first_length)/edited_fps
                         + .947368*max(0,frame-first_length)/edited_fps)
            source_frame=int(np.floor(source_time*source_fps+1e-8))
            edited.append(dict(
                edited_frame_number=frame,
                edited_timestamp_ms=frame*1000/edited_fps,
                edited_phash=hashes[source_frame],
                top_n_matches=[dict(original_video_name='s',
                                    original_frame_number=source_frame,
                                    original_timestamp_ms=source_frame*1000/source_fps,
                                    phash_distance=0)]))
        result=align_segments(pd.DataFrame(edited),originals)
        covered=sum(result.edited_end_frame-result.edited_start_frame+1)
        self.assertEqual(covered,total)
        self.assertLess(result.speed.min(),.85)
        self.assertGreater(result.speed.max(),.92)

    def test_pixel_fallback_recovers_phash_unstable_freeze(self):
        rng=np.random.default_rng(91)
        source_pixels=rng.integers(0,256,(40,64,64),dtype=np.uint8)
        query_pixels=np.repeat(source_pixels[17:18],24,axis=0)
        source_hashes=[row.tobytes().hex() for row in rng.integers(0,256,(40,32),dtype=np.uint8)]
        originals=pd.DataFrame(dict(original_video_name=['s']*40,
                                    original_frame_number=range(40),
                                    original_timestamp_ms=np.arange(40)*1000/30,
                                    original_phash=source_hashes))
        edited=pd.DataFrame(dict(edited_frame_number=range(24),
                                 edited_timestamp_ms=np.arange(24)*1000/60,
                                 edited_phash=['0'*64]*24,
                                 top_n_matches=[[] for _ in range(24)]))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'edited').mkdir();(root/'s').mkdir()
            np.save(root/'edited'/'compact_frames.npy',query_pixels)
            np.save(root/'s'/'compact_frames.npy',source_pixels)
            result=align_segments(edited,originals,method='affine_pixels',
                                  output_dir=root,edited_name='edited')
        self.assertEqual(result.edited_start_frame.tolist(),[0])
        self.assertEqual(result.edited_end_frame.tolist(),[23])
        self.assertAlmostEqual(result.speed.iloc[0],0)
        self.assertEqual(result.original_start_frame.iloc[0],17)

    def test_pixel_verified_three_frame_flash_is_preserved(self):
        rng=np.random.default_rng(333)
        source_pixels=rng.integers(0,256,(80,64,64),dtype=np.uint8)
        source_hashes=[row.tobytes().hex() for row in rng.integers(0,256,(80,32),dtype=np.uint8)]
        originals=pd.DataFrame(dict(original_video_name=['s']*80,
                                    original_frame_number=range(80),
                                    original_timestamp_ms=np.arange(80)*1000/30,
                                    original_phash=source_hashes))
        source_ids=([frame//2 for frame in range(20)] + [70]*3
                    + [10+frame//2 for frame in range(20)] + [70]*12)
        edited=[]
        for frame,source_frame in enumerate(source_ids):
            pixel_only=20 <= frame < 23 or frame >= 43
            edited.append(dict(
                edited_frame_number=frame,edited_timestamp_ms=frame*1000/60,
                edited_phash='0'*64 if pixel_only else source_hashes[source_frame],
                top_n_matches=[] if pixel_only else [dict(
                    original_video_name='s',original_frame_number=source_frame,
                    original_timestamp_ms=source_frame*1000/30,phash_distance=0)]))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'edited').mkdir();(root/'s').mkdir()
            np.save(root/'edited'/'compact_frames.npy',source_pixels[source_ids])
            np.save(root/'s'/'compact_frames.npy',source_pixels)
            result=align_segments(pd.DataFrame(edited),originals,method='affine_pixels',
                                  output_dir=root,edited_name='edited')
        self.assertEqual(result.edited_start_frame.tolist(),[0,20,23,43])
        self.assertEqual(result.edited_end_frame.tolist(),[19,22,42,54])
        self.assertEqual(result.original_start_frame.tolist(),[0,70,10,70])
        self.assertEqual(result.speed.tolist(),[1,0,1,0])

    def test_reject_invalid_algorithm_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'settings.toml'
            path.write_text('[paths]\nedited_video="a.mp4"\n[step3]\ntop_k=0\n')
            with self.assertRaises(ValueError):load_settings(path)
            path.write_text('[paths]\nedited_video="a.mp4"\n[step1]\nframe_storage="png"\n')
            with self.assertRaises(ValueError):load_settings(path)

    def test_unique_single_and_three_frame_cutaways_need_no_long_hypothesis(self):
        rng = np.random.default_rng(624)
        pixels = rng.integers(0, 256, (90, 64, 64), dtype=np.uint8)
        hashes = [row.tobytes().hex() for row in rng.integers(0, 256, (90, 32), dtype=np.uint8)]
        originals = pd.DataFrame(dict(original_video_name=['s']*90, original_frame_number=range(90),
                                     original_timestamp_ms=np.arange(90)*1000/30, original_phash=hashes))
        for length in (1, 3):
            ids = list(range(20)) + [75]*length + list(range(20, 40))
            edited = pd.DataFrame([dict(edited_frame_number=i, edited_timestamp_ms=i*1000/30,
                                       edited_phash=hashes[index], top_n_matches=[dict(
                                           original_video_name='s', original_frame_number=index,
                                           original_timestamp_ms=index*1000/30, phash_distance=0)])
                                   for i, index in enumerate(ids)])
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'s').mkdir(); (root/'q').mkdir()
                np.save(root/'s'/'compact_frames.npy', pixels)
                np.save(root/'q'/'compact_frames.npy', pixels[ids])
                result = align_segments(edited, originals, 'affine_pixels', root, 'q')
            self.assertEqual(sum(result.edited_end_frame-result.edited_start_frame+1), len(ids))
            short = result[result.edited_start_frame == 20]
            self.assertEqual(short.original_start_frame.tolist(), [75])
            self.assertEqual(short.edited_end_frame.tolist(), [20+length-1])

    def test_flat_frame_pixel_search_runs_even_when_wrong_phash_match_is_exact(self):
        pixels = np.zeros((50, 64, 64), dtype=np.uint8)
        pixels[35:] = 255
        originals = pd.DataFrame(dict(original_video_name=['s']*50, original_frame_number=range(50),
                                     original_timestamp_ms=np.arange(50)*40, original_phash=['0'*64]*50))
        candidates = [dict(original_video_name='s', original_frame_number=i,
                           original_timestamp_ms=i*40, phash_distance=0) for i in range(3)]
        edited = pd.DataFrame(dict(edited_frame_number=range(16), edited_timestamp_ms=np.arange(16)*40,
                                  edited_phash=['0'*64]*16, top_n_matches=[candidates for _ in range(16)]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'s').mkdir(); (root/'q').mkdir()
            np.save(root/'s'/'compact_frames.npy', pixels)
            np.save(root/'q'/'compact_frames.npy', np.full((16,64,64),255,dtype=np.uint8))
            result = align_segments(edited, originals, 'affine_pixels', root, 'q')
        self.assertEqual(sum(result.edited_end_frame-result.edited_start_frame+1), 16)
        self.assertTrue((result.original_start_frame >= 35).all())


if __name__=='__main__':
    unittest.main()
