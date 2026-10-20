"""Bounded, lossless source-analysis cache. Never cache an RT decision.

Content hashes cover input files, intrinsics, range/model and local Python code.
NPZ arrays + JSON metadata only: no pickle or executable objects are loaded.
Cache errors fall back to ordinary analysis; source files are never modified.
"""
from collections import OrderedDict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from threading import RLock
from time import perf_counter

import numpy as np
import cv2
import scipy
from app_logging import get_logger
from app_paths import output_dir

LOGGER = get_logger('analysis_cache')
ROOT = Path(__file__).resolve().parent


class AnalysisCache:
    def __init__(self, folder=None, memory_items=12, disk_bytes=1024**3):
        self.folder = Path(folder) if folder is not None else output_dir('cache')/'nsl_analysis'
        self.memory_items, self.disk_bytes = memory_items, disk_bytes
        self.memory = OrderedDict()
        self.lock = RLock()
        self.hits = self.misses = 0

    def key(self, pair, color_cal, z_min, z_max, geometry):
        digest = sha256(json.dumps([2, pair['timestamp'], z_min, z_max, geometry,
            sys.version, np.__version__, cv2.__version__, scipy.__version__]).encode())
        files = [pair['color'], pair['cloud'], Path(color_cal)/'Intrinsic.txt',
                 Path(color_cal)/'Distortion.txt']
        if geometry == 'wf_range':
            files.append(pair['range'])
        # A code edit invalidates old results, including imported geometry rules.
        for path in files + sorted(ROOT.glob('*.py')):
            path = Path(path).resolve()
            digest.update(str(path).encode())
            digest.update(sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def analyze(self, pair, color_cal, z_min, z_max, geometry, analyzer):
        started = perf_counter()
        key = None
        try:
            key = self.key(pair, color_cal, z_min, z_max, geometry)
            with self.lock:
                if key in self.memory:
                    frame = self.memory.pop(key)
                    self.memory[key] = frame
                    self.hits += 1
                    return deepcopy(frame)
                path = self.folder/f'{key}.npz'
                if path.is_file():
                    frame = self.read(path)
                    path.touch()  # LRU on this application's cache files only.
                    self.remember(key, frame)
                    self.hits += 1
                    LOGGER.info('analysis cache hit stamp=%s elapsed=%.3fs', pair['timestamp'], perf_counter()-started)
                    return frame
        except Exception:
            LOGGER.warning('analysis cache unavailable; reanalyzing stamp=%s', pair['timestamp'], exc_info=True)
        self.misses += 1
        frame = analyzer(pair, color_cal, z_min, z_max, geometry)
        try:
            # Do not store a result under an input that changed during analysis.
            if key and key == self.key(pair, color_cal, z_min, z_max, geometry):
                with self.lock:
                    self.remember(key, frame)
                    self.write(self.folder/f'{key}.npz', frame)
                    self.trim()
        except Exception:
            LOGGER.warning('analysis cache save skipped stamp=%s', pair['timestamp'], exc_info=True)
        return frame

    def remember(self, key, frame):
        # Callers freely change RT-derived fields. Keep the raw analysis isolated.
        self.memory[key] = deepcopy(frame)
        self.memory.move_to_end(key)
        while len(self.memory) > self.memory_items:
            self.memory.popitem(last=False)

    @staticmethod
    def write(path, frame):
        arrays = {}
        def encode(value):
            if isinstance(value, np.ndarray):
                name = f'a{len(arrays)}'
                arrays[name] = value
                return {'array': name}
            if isinstance(value, Path):
                return {'path': str(value)}
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            return value
        plane = frame.get('source_plane')
        index = (next(i for i, p in enumerate(frame['source_candidates']) if p is plane)
                 if plane is not None else None)
        metadata = json.dumps(dict(frame=encode(dict(frame, source_plane=None)),
                                   source_index=index), ensure_ascii=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, metadata=metadata, **arrays)
        temporary.replace(path)

    @staticmethod
    def read(path):
        with np.load(path, allow_pickle=False) as data:
            def decode(value):
                if isinstance(value, dict):
                    if set(value) == {'array'}:
                        return data[value['array']]
                    if set(value) == {'path'}:
                        return Path(value['path'])
                    return {k: decode(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [decode(v) for v in value]
                return value
            metadata = json.loads(str(data['metadata']))
            frame = decode(metadata['frame'])
        # Equal masks need not have equal boundary evidence; preserve exact index.
        frame['pcd_shape'] = tuple(frame['pcd_shape'])
        if metadata['source_index'] is not None:
            frame['source_plane'] = frame['source_candidates'][metadata['source_index']]
        return frame

    def trim(self):
        files = sorted(self.folder.glob('*.npz'), key=lambda p: p.stat().st_mtime_ns)
        total = sum(p.stat().st_size for p in files)
        for path in files:
            if total <= self.disk_bytes:
                break
            total -= path.stat().st_size
            path.unlink()  # Only regenerable files in our dedicated cache folder.
