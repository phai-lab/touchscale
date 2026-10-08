from collections.abc import Iterator, Mapping, Sequence
import dataclasses
import functools
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ModuleNotFoundError:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

try:
    from lerobot.common.datasets.video_utils import decode_video_frames, get_safe_default_codec
except ModuleNotFoundError:
    from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec

import n0vtla.models.model as _model
import n0vtla.training.config as _config
from n0vtla.training.droid_rlds_dataset import DroidRldsDataset
import n0vtla.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
_ORIGINAL_LEROBOT_DECODE_VIDEO_FRAMES = decode_video_frames


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


@dataclasses.dataclass(frozen=True)
class _InjectConstants:
    """Merge constant key/values into every sample (e.g. the per-repo norm_group_id tag)."""

    constants: dict

    def __call__(self, data: dict) -> dict:
        data.update(self.constants)
        return data


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class IndexFilteredDataset(Dataset[T_co]):
    """Wrap a dataset and expose only the selected indices."""

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]):
        self._dataset = dataset
        self._indices = tuple(int(index) for index in indices)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._dataset[self._indices[index.__index__()]]

    def __len__(self) -> int:
        return len(self._indices)


class LocalLeRobotV3Dataset(Dataset[dict]):
    """Minimal local dataset wrapper for LeRobot v3 roots used by n0vtla training."""

    def __init__(
        self,
        root: str | pathlib.Path,
        *,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        video_backend: str | None = None,
        include_videos: bool = True,
        video_keys: Sequence[str] | None = None,
        precomputed_video_root: str | os.PathLike[str] | None = None,
    ):
        import collections

        self.root = pathlib.Path(root)
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend or get_safe_default_codec()
        self.info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.tasks = _load_tasks_v3(self.root)
        self.episodes = _load_episodes_v3(self.root)
        available_video_keys = [key for key, ft in self.info["features"].items() if ft["dtype"] == "video"]
        if include_videos:
            if video_keys is None:
                self.video_keys = available_video_keys
            else:
                requested_video_keys = [key for key in video_keys if key in available_video_keys]
                missing_video_keys = [key for key in video_keys if key not in available_video_keys]
                if missing_video_keys:
                    # Mixed-embodiment canonical: single-arm repos genuinely lack right_wrist views
                    # (right_wrist_view / *_tactile), etc. The repack references the FULL canonical
                    # key set, so a repo missing some is EXPECTED — load what's present; the input
                    # transform (CanonicalTactileInputs) placeholders the absent views (mask=False).
                    logging.debug("Repo %s lacks some requested video keys (loading present only): %s", self.root, missing_video_keys)
                self.video_keys = requested_video_keys
        else:
            self.video_keys = []
        self.camera_keys = list(self.video_keys)
        self.fps = float(self.info["fps"])
        self._chunks_size = int(self.info.get("chunks_size", 1000))  # v2.1 video/data chunk grouping
        self.precomputed_videos = _load_precomputed_video_caches(self.root, precomputed_video_root, self.video_keys)
        self.total_episodes = len(self.episodes)
        self.episode_data_index = {
            ep_idx: (int(row["dataset_from_index"]), int(row["dataset_to_index"])) for ep_idx, row in self.episodes.items()
        }
        # v2.1 (ONE parquet per episode, data_path carries {episode_index}) → LAZY per-episode
        # reads. A large multi-repo corpus cannot be eager-loaded: on network storage that costs
        # hours of latency and well over 100 GB of RAM.
        # v3 (episodes PACKED into file-YYY.parquet, data_path carries {file_index}) → EAGER
        # whole-repo load. Per-episode lazy paths do not exist for packed files, and v3 datasets
        # here are single small repos where eager loading is fine. Keeping this branch eager also
        # leaves the v3 path byte-identical to the pre-lazy implementation.
        data_path_tmpl = str(self.info.get("data_path", ""))
        self._lazy = "{episode_index" in data_path_tmpl
        if self._lazy:
            self.num_frames = int(self.info["total_frames"])
            # frame -> episode routing: sorted (from_index, ep_idx) for bisect in __getitem__.
            self._ep_bounds = sorted((fr, ep) for ep, (fr, _to) in self.episode_data_index.items())
            self._data_path_tmpl = data_path_tmpl
            self._ep_cache: collections.OrderedDict = collections.OrderedDict()  # ep_idx -> {col: tensor|list}
            # Kept TINY on purpose: this cache is PER sub-dataset, and a canonical ConcatDataset has
            # 577 of them, each forked across DataLoader workers (COW). 8×577×8 workers ≈ 30GB still
            # OOM'd 64GB at 22%. A single __getitem__'s delta (baseline/current/future) all resolve
            # to ONE episode (read once), and shuffled access has ~0 cross-sample reuse — 2 suffices.
            self._ep_cache_max = 2
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq

            data_tables = [pq.read_table(parquet_path) for parquet_path in sorted((self.root / "data").rglob("*.parquet"))]
            if not data_tables:
                raise ValueError(f"No parquet files found under {self.root / 'data'}")
            data_table = pa.concat_tables(data_tables) if len(data_tables) > 1 else data_tables[0]
            self._columns = {}
            for name in data_table.column_names:
                column = data_table[name]
                if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
                    self._columns[name] = column.to_pylist()
                    continue
                values = np.asarray(
                    column.to_pylist()
                    if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type)
                    else column.to_numpy(zero_copy_only=False)
                )
                if np.issubdtype(values.dtype, np.floating):
                    values = values.astype(np.float32, copy=False)
                self._columns[name] = torch.from_numpy(values)
            self.num_frames = len(next(iter(self._columns.values())))
            if self.num_frames != int(self.info["total_frames"]):
                raise ValueError(
                    f"Frame count mismatch for {self.root}: info has {self.info['total_frames']}, parquet has {self.num_frames}"
                )
        self.delta_indices = None
        if delta_timestamps is not None:
            self.delta_indices = {
                key: [int(round(timestamp * self.fps)) for timestamp in timestamps]
                for key, timestamps in delta_timestamps.items()
            }

        for key, cache in self.precomputed_videos.items():
            if cache.shape[0] != self.num_frames:
                raise ValueError(
                    f"Precomputed video cache for {key} in {self.root} has {cache.shape[0]} frames,"
                    f" expected {self.num_frames}"
                )

    def __len__(self) -> int:
        return self.num_frames

    def _frame_to_ep(self, idx: int) -> int:
        """Global frame idx -> episode index, via bisect on sorted (from_index, ep) bounds."""
        import bisect

        pos = bisect.bisect_right(self._ep_bounds, (idx, self.total_episodes + 1)) - 1
        return self._ep_bounds[pos][1]

    def _episode_columns(self, ep_idx: int) -> dict:
        """Lazily read + LRU-cache one episode's data parquet as {col: tensor|list}. This is the
        core of the lazy design: only the touched episodes' parquet is read, amortising network
        storage latency across DataLoader workers, instead of eager-loading every repo at build
        time."""
        cached = self._ep_cache.get(ep_idx)
        if cached is not None:
            self._ep_cache.move_to_end(ep_idx)
            return cached
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.root / self._data_path_tmpl.format(
            episode_chunk=ep_idx // self._chunks_size, episode_index=ep_idx
        )
        table = pq.read_table(path)
        cols: dict = {}
        for name in table.column_names:
            column = table[name]
            if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
                cols[name] = column.to_pylist()
                continue
            values = np.asarray(
                column.to_pylist()
                if pa.types.is_fixed_size_list(column.type) or pa.types.is_list(column.type)
                else column.to_numpy(zero_copy_only=False)
            )
            if np.issubdtype(values.dtype, np.floating):
                values = values.astype(np.float32, copy=False)
            cols[name] = torch.from_numpy(values)
        self._ep_cache[ep_idx] = cols
        if len(self._ep_cache) > self._ep_cache_max:
            self._ep_cache.popitem(last=False)
        return cols

    def _global_col(self, key: str, global_idx: int):
        """Fetch one (key, global frame idx) value through the per-episode lazy cache."""
        ep = self._frame_to_ep(global_idx)
        local = global_idx - self.episode_data_index[ep][0]
        return self._episode_columns(ep)[key][local]

    def _get_item(self, idx: int) -> dict:
        if not self._lazy:  # v3 eager: original behavior
            return {key: values[idx] for key, values in self._columns.items()}
        ep_idx = self._frame_to_ep(idx)
        local = idx - self.episode_data_index[ep_idx][0]
        cols = self._episode_columns(ep_idx)
        return {key: col[local] for key, col in cols.items()}

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep_start, ep_end = self.episode_data_index[ep_idx]
        assert self.delta_indices is not None
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start) | (idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_data_columns(self, query_indices: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        if not self._lazy:  # v3 eager: original behavior + missing-column skip (mirrors lazy)
            return {
                key: self._columns[key][q_idx]
                for key, q_idx in query_indices.items()
                if key not in self.video_keys and key in self._columns
            }
        out: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.video_keys:
                continue
            # extra_delta_timestamps references ALL tactile keys, which are VIDEO features (absent
            # from the data parquet) — and for single-arm repos some views don't exist at all. Skip
            # any key not present as a data column this episode; tactile video delta is handled by
            # _query_videos (present video_keys), the rest are placeholdered by CanonicalTactileInputs.
            ep = self._frame_to_ep(int(q_idx[0]))
            cols = self._episode_columns(ep)
            if key not in cols:
                continue
            ep_start = self.episode_data_index[ep][0]
            vals = [cols[key][int(gi) - ep_start] for gi in q_idx]
            out[key] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
        return out

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.video_keys:
            if query_indices is not None and key in query_indices:
                if not self._lazy:  # v3 eager: original behavior
                    query_timestamps[key] = self._columns["timestamp"][query_indices[key]].tolist()
                else:
                    query_timestamps[key] = [float(self._global_col("timestamp", int(gi))) for gi in query_indices[key]]
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_videos(
        self,
        idx: int,
        query_timestamps: dict[str, list[float]],
        ep_idx: int,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, torch.Tensor]:
        row = self.episodes[ep_idx]
        item = {}
        # v2.1 (jsonl-meta): the episode row has NO videos/<key>/* fields; each episode is one mp4 at
        # videos/chunk-{ep//chunks_size}/<key>/episode_{ep:06d}.mp4, and timestamps are episode-relative
        # (base 0). v3 (parquet-meta): videos/<key>/chunk_index+file_index+from_timestamp, many episodes
        # packed per file-YYY.mp4. Detect by the presence of the v3 field on the episode row.
        is_v21 = bool(self.video_keys) and f"videos/{self.video_keys[0]}/chunk_index" not in row
        for vid_key, query_ts in query_timestamps.items():
            if vid_key in self.precomputed_videos:
                frame_indices = query_indices[vid_key] if query_indices is not None and vid_key in query_indices else [idx]
                item[vid_key] = _load_precomputed_video_frames(self.precomputed_videos[vid_key], frame_indices)
                continue
            if is_v21:
                chunk_index = ep_idx // self._chunks_size
                base_timestamp = 0.0
                video_path = self.root / "videos" / f"chunk-{chunk_index:03d}" / vid_key / f"episode_{ep_idx:06d}.mp4"
            else:
                chunk_index = int(row[f"videos/{vid_key}/chunk_index"])
                file_index = int(row[f"videos/{vid_key}/file_index"])
                base_timestamp = float(row[f"videos/{vid_key}/from_timestamp"])
                video_path = self.root / "videos" / vid_key / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            absolute_ts = [base_timestamp + ts for ts in query_ts]
            if self.video_backend == "pyav":
                frames = _decode_video_frames_precise_pyav(video_path, absolute_ts, self.tolerance_s)
            else:
                frames = decode_video_frames(video_path, absolute_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)
        return item

    def __getitem__(self, idx: int) -> dict:
        item = self._get_item(idx)
        ep_idx = int(item["episode_index"].item())

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_data_columns(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if self.video_keys:
            current_ts = float(item["timestamp"].item())
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(idx, query_timestamps, ep_idx, query_indices)
            item = {**video_frames, **item}

        task_idx = int(item["task_index"].item())
        item["task"] = self.tasks[task_idx]
        return item


def _is_local_lerobot_v3_root(repo_id: str | os.PathLike[str]) -> bool:
    """A local LeRobot root we can read directly — v3 (parquet meta) OR v2.1 (jsonl meta).

    Both store frame data as data/**/*.parquet with a meta/info.json; they differ only in the
    task/episode metadata format (v3: tasks.parquet + meta/episodes/ dir; v2.1: tasks.jsonl +
    episodes.jsonl). _load_tasks_v3 / _load_episodes_v3 both handle either, so accept either —
    this keeps canonical umi+non_umi (v2.1) on the LOCAL path instead of falling through to the
    HF-hub branch (which would treat the placeholder repo_id as a HF dataset and time out offline).
    """
    repo_path = pathlib.Path(repo_id)
    if not (repo_path.is_dir() and (repo_path / "data").is_dir() and (repo_path / "meta" / "info.json").is_file()):
        return False
    has_tasks = (repo_path / "meta" / "tasks.parquet").is_file() or (repo_path / "meta" / "tasks.jsonl").is_file()
    has_eps = (repo_path / "meta" / "episodes").is_dir() or (repo_path / "meta" / "episodes.jsonl").is_file()
    return has_tasks and has_eps


def _load_tasks_v3(repo_path: pathlib.Path) -> dict[int, str]:
    tasks_parquet = repo_path / "meta" / "tasks.parquet"
    if tasks_parquet.is_file():
        import pyarrow.parquet as pq

        rows = pq.read_table(tasks_parquet).to_pylist()
        tasks = {}
        for row in rows:
            task_index = int(row["task_index"])
            tasks[task_index] = row.get("task", row.get("__index_level_0__"))
        return tasks
    # v2.1 fallback: meta/tasks.jsonl, one {"task", "task_index"} object per line.
    tasks = {}
    with (repo_path / "meta" / "tasks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    return tasks


def _load_precomputed_video_caches(
    repo_path: pathlib.Path,
    precomputed_video_root: str | os.PathLike[str] | None,
    video_keys: Sequence[str],
) -> dict[str, np.ndarray]:
    if precomputed_video_root is None:
        return {}

    cache_root = pathlib.Path(precomputed_video_root) / repo_path.name
    caches: dict[str, np.ndarray] = {}
    for key in video_keys:
        cache_path = cache_root / f"{key}.npy"
        if not cache_path.is_file():
            continue
        caches[key] = np.load(cache_path, mmap_mode="r")
    if caches:
        logging.info("Loaded %d precomputed video caches for %s from %s", len(caches), repo_path, cache_root)
    return caches


def _load_precomputed_video_frames(cache: np.ndarray, frame_indices: Sequence[int]) -> torch.Tensor:
    indices = np.asarray(frame_indices, dtype=np.int64)
    frames = np.take(cache, indices, axis=0)
    if frames.ndim == 3:
        frames = np.moveaxis(frames, -1, 0)
        return torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0

    if frames.ndim != 4:
        raise ValueError(f"Expected precomputed video cache with rank 3 or 4 after indexing, got {frames.shape}")
    frames = np.moveaxis(frames, -1, 1)
    frames = torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0
    return frames.squeeze(0) if frames.shape[0] == 1 else frames


def _load_episodes_v3(repo_path: pathlib.Path) -> dict[int, dict]:
    episodes_dir = repo_path / "meta" / "episodes"
    if episodes_dir.is_dir():
        import pyarrow.parquet as pq

        rows = []
        for parquet_path in sorted(episodes_dir.rglob("*.parquet")):
            rows.extend(pq.read_table(parquet_path).to_pylist())
        return {int(row["episode_index"]): row for row in sorted(rows, key=lambda row: row["episode_index"])}
    # v2.1 fallback: meta/episodes.jsonl has {episode_index, length} but NO dataset_from/to_index.
    # Reconstruct each episode's global frame range by cumulative sum of lengths in episode_index
    # order. This MUST match the concat order of sorted(data/**/*.parquet) in LocalLeRobotV3Dataset
    # (both ascend by episode_index: data files are episode_XXXXXX.parquet, one per episode).
    rows = []
    with (repo_path / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["episode_index"]))
    episodes: dict[int, dict] = {}
    cursor = 0
    for row in rows:
        ep_idx, length = int(row["episode_index"]), int(row["length"])
        episodes[ep_idx] = {**row, "dataset_from_index": cursor, "dataset_to_index": cursor + length}
        cursor += length
    return episodes


def _load_episode_lengths(repo_path: pathlib.Path) -> list[int]:
    if (repo_path / "meta" / "episodes.jsonl").is_file():
        episodes_path = repo_path / "meta" / "episodes.jsonl"
        with episodes_path.open(encoding="utf-8") as f:
            return [int(json.loads(line)["length"]) for line in f]

    if (repo_path / "meta" / "episodes").is_dir():
        return [int(row["length"]) for _, row in sorted(_load_episodes_v3(repo_path).items())]

    episodes_path = repo_path / "meta" / "episodes.jsonl"
    with episodes_path.open(encoding="utf-8") as f:
        return [int(json.loads(line)["length"]) for line in f]


def _iter_string_leaves(tree: object) -> Iterator[str]:
    if isinstance(tree, str):
        yield tree
        return
    if isinstance(tree, Mapping):
        for value in tree.values():
            yield from _iter_string_leaves(value)
        return
    if isinstance(tree, Sequence) and not isinstance(tree, str | bytes):
        for value in tree:
            yield from _iter_string_leaves(value)


def _extract_referenced_video_keys(
    repack_transforms: _transforms.Group,
    available_video_keys: Sequence[str],
) -> list[str]:
    available = set(available_video_keys)
    referenced: list[str] = []
    seen: set[str] = set()
    for transform in repack_transforms.inputs:
        if not isinstance(transform, _transforms.RepackTransform):
            continue
        for key in _iter_string_leaves(transform.structure):
            if key in available and key not in seen:
                referenced.append(key)
                seen.add(key)
    return referenced


@functools.lru_cache(maxsize=256)
def _pyav_video_stream_info(video_path: str) -> tuple[float, float]:
    import av

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        fps = float(stream.average_rate) if stream.average_rate is not None else 30.0
    return time_base, fps


def _decode_video_frames_precise_pyav(
    video_path: pathlib.Path | str,
    timestamps: Sequence[float],
    tolerance_s: float,
) -> torch.Tensor:
    import av

    path_str = str(video_path)
    query_ts = sorted({float(ts) for ts in timestamps})
    time_base, fps = _pyav_video_stream_info(path_str)

    cluster_gap_s = 2.0
    clusters: list[list[float]] = [[query_ts[0]]]
    for ts in query_ts[1:]:
        if ts - clusters[-1][-1] > cluster_gap_s:
            clusters.append([ts])
        else:
            clusters[-1].append(ts)

    decoded_frames: list[np.ndarray] = []
    decoded_ts: list[float] = []
    with av.open(path_str) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for cluster in clusters:
            c_first, c_last = cluster[0], cluster[-1]
            # torchvision's pyav backend can seek to the next keyframe and miss the target frame.
            # Seek earlier and decode forward until we cover the cluster's timestamps.
            seek_margin_s = max(1.1, 2.0 / max(fps, 1.0))
            while True:
                seek_ts = max(0.0, c_first - seek_margin_s)
                local_frames: list[np.ndarray] = []
                local_ts: list[float] = []
                container.seek(int(seek_ts / time_base), stream=stream, any_frame=False, backward=True)
                for frame in container.decode(stream):
                    if frame.pts is None:
                        continue
                    frame_ts = float(frame.pts * stream.time_base)
                    if frame_ts + tolerance_s < c_first:
                        continue
                    local_frames.append(frame.to_ndarray(format="rgb24"))
                    local_ts.append(frame_ts)
                    if frame_ts >= c_last:
                        break
                if local_ts and (local_ts[0] <= c_first + tolerance_s or seek_ts == 0.0):
                    decoded_frames.extend(local_frames)
                    decoded_ts.extend(local_ts)
                    break
                seek_margin_s *= 2.0
                if seek_ts == 0.0:
                    decoded_frames.extend(local_frames)
                    decoded_ts.extend(local_ts)
                    break

    if not decoded_ts:
        raise RuntimeError(f"Unable to decode frames for {path_str} at timestamps {query_ts}")

    loaded_ts = torch.tensor(decoded_ts, dtype=torch.float32)
    query_tensor = torch.tensor(list(timestamps), dtype=torch.float32)
    dist = torch.cdist(query_tensor[:, None], loaded_ts[:, None], p=1)
    min_dist, argmin = dist.min(1)
    tol = max(tolerance_s, 0.5 / max(fps, 1.0))
    is_within_tol = min_dist < tol
    if not bool(is_within_tol.all()):
        worst = float(min_dist.max())
        if worst > 2.0:
            raise AssertionError(
                f"Query timestamps violate even the 2s fallback tolerance ({min_dist[~is_within_tol]})."
                f"\nqueried: {query_tensor}\nloaded: {loaded_ts}\nvideo: {path_str}"
            )
        warned = getattr(_decode_video_frames_precise_pyav, "_warned", None)
        if warned is None:
            warned = set()
            _decode_video_frames_precise_pyav._warned = warned  # noqa: SLF001
        if path_str not in warned:
            warned.add(path_str)
            logging.warning(
                "video shorter than parquet (nearest-frame fallback, max gap %.3fs): %s", worst, path_str
            )

    closest_frames = torch.stack(
        [torch.from_numpy(np.ascontiguousarray(decoded_frames[idx.item()])).permute(2, 0, 1) for idx in argmin]
    ).to(torch.float32) / 255.0
    return closest_frames


def _decode_video_frames_precise_pyav_compat(
    video_path: pathlib.Path | str,
    timestamps: Sequence[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    if backend == "pyav":
        return _decode_video_frames_precise_pyav(video_path, timestamps, tolerance_s)
    return _ORIGINAL_LEROBOT_DECODE_VIDEO_FRAMES(video_path, timestamps, tolerance_s, backend)


def _enable_precise_pyav_for_lerobot() -> None:
    if getattr(lerobot_dataset, "_n0vtla_precise_pyav_enabled", False):
        return
    lerobot_dataset.decode_video_frames = _decode_video_frames_precise_pyav_compat
    setattr(lerobot_dataset, "_n0vtla_precise_pyav_enabled", True)


def _load_excluded_episode_indices(repo_path: pathlib.Path, total_episodes: int) -> set[int]:
    metadata_path = repo_path / "metadata.json"
    if not metadata_path.is_file():
        logging.info("Anomaly filtering requested for %s, but metadata.json was not found; skipping.", repo_path)
        return set()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    anomaly_entries = metadata.get("anomaly_episodes") or []
    raw_indices: list[int] = []
    for entry in anomaly_entries:
        if isinstance(entry, int):
            raw_indices.append(entry)
            continue
        if isinstance(entry, dict):
            episode_index = entry.get("episodeIndex", entry.get("episode_index"))
            if episode_index is not None:
                raw_indices.append(int(episode_index))

    if not raw_indices:
        return set()

    # Current ARX metadata stores episode indices as 1-based integers, but keep a 0-based fallback for safety.
    if 0 in raw_indices:
        normalized_indices = {index for index in raw_indices if 0 <= index < total_episodes}
    else:
        normalized_indices = {index - 1 for index in raw_indices if 1 <= index <= total_episodes}

    dropped_indices = set(raw_indices) - (
        normalized_indices if 0 in raw_indices else {index + 1 for index in normalized_indices}
    )
    if dropped_indices:
        logging.warning(
            "Ignoring out-of-range anomaly episode indices for %s: %s",
            repo_path,
            sorted(dropped_indices),
        )

    return normalized_indices


def _build_non_anomalous_indices(repo_ids: Sequence[str]) -> list[int]:
    kept_indices: list[int] = []
    global_frame_offset = 0
    total_removed_frames = 0
    total_removed_episodes = 0

    for repo_id in repo_ids:
        repo_path = pathlib.Path(repo_id)
        if not repo_path.is_dir():
            logging.warning(
                "Anomaly filtering only supports local LeRobot dataset roots. Skipping filtering for %s.",
                repo_id,
            )
            return []

        episode_lengths = _load_episode_lengths(repo_path)
        excluded_episode_indices = _load_excluded_episode_indices(repo_path, len(episode_lengths))

        repo_frame_offset = global_frame_offset
        removed_frames = 0
        for episode_index, episode_length in enumerate(episode_lengths):
            if episode_index in excluded_episode_indices:
                removed_frames += episode_length
            else:
                kept_indices.extend(range(global_frame_offset, global_frame_offset + episode_length))
            global_frame_offset += episode_length

        total_removed_frames += removed_frames
        total_removed_episodes += len(excluded_episode_indices)
        logging.info(
            "Filtered anomaly episodes for %s: removed %d/%d episodes and %d/%d frames.",
            repo_id,
            len(excluded_episode_indices),
            len(episode_lengths),
            removed_frames,
            global_frame_offset - repo_frame_offset,
        )

    logging.info(
        "Anomaly filtering removed %d episodes and %d frames across %d dataset roots.",
        total_removed_episodes,
        total_removed_frames,
        len(repo_ids),
    )
    return kept_indices


def _disable_lerobot_video_loading(dataset: Dataset) -> None:
    def _disable_for_meta(meta) -> None:
        for feature in meta.info["features"].values():
            if feature.get("dtype") == "video":
                feature["dtype"] = "disabled_video"

    if hasattr(dataset, "_datasets"):
        for inner_dataset in dataset._datasets:
            _disable_for_meta(inner_dataset.meta)
    elif hasattr(dataset, "meta"):
        _disable_for_meta(dataset.meta)


def _build_delta_timestamps(
    data_config: _config.DataConfig,
    fps: float,
    action_horizon: int,
    *,
    include_videos: bool,
) -> dict[str, list[float]]:
    delta_timestamps = {key: [t / fps for t in range(action_horizon)] for key in data_config.action_sequence_keys}
    if data_config.extra_delta_timestamps is None or not include_videos:
        return delta_timestamps

    overlap = set(delta_timestamps) & set(data_config.extra_delta_timestamps)
    if overlap:
        raise ValueError(f"Overlapping delta timestamp keys are not supported: {tuple(sorted(overlap))}")

    for key, timestamps in data_config.extra_delta_timestamps.items():
        delta_timestamps[key] = list(timestamps)
    return delta_timestamps


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    include_videos: bool = True,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    repo_ids = tuple(data_config.repo_ids) if data_config.repo_ids else ()
    if repo_id is None and not repo_ids:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    task_mapping: dict[int, str] | None = None
    video_backend = "pyav"
    try:
        from torchcodec.decoders import VideoDecoder  # noqa: F401

        video_backend = "torchcodec"
    except Exception:
        logging.info("torchcodec backend unavailable, falling back to pyav for LeRobot video decoding")
        _enable_precise_pyav_for_lerobot()

    local_roots = repo_ids if repo_ids else ((repo_id,) if repo_id is not None else ())
    if local_roots and all(_is_local_lerobot_v3_root(root) for root in local_roots):
        info = json.loads((pathlib.Path(local_roots[0]) / "meta" / "info.json").read_text(encoding="utf-8"))
        task_mapping = _load_tasks_v3(pathlib.Path(local_roots[0]))
        fps = float(info["fps"])
        # UNION of video keys across ALL roots: mixed-embodiment repos differ in available views
        # (single-arm lacks right_wrist*; dual-arm lacks second_third_view). Filtering by root[0]
        # alone silently drops keys corpus-wide depending on the glob sort order — every repo that
        # HAS the key would then train on a zero placeholder. Per-repo missing keys are already
        # handled inside LocalLeRobotV3Dataset (load present only).
        available_video_keys: list[str] = []
        _seen_vk: set[str] = set()
        for _root in local_roots:
            _info_i = info if _root == local_roots[0] else json.loads(
                (pathlib.Path(_root) / "meta" / "info.json").read_text(encoding="utf-8")
            )
            for key, ft in _info_i["features"].items():
                if ft.get("dtype") == "video" and key not in _seen_vk:
                    _seen_vk.add(key)
                    available_video_keys.append(key)
        requested_video_keys = (
            _extract_referenced_video_keys(data_config.repack_transforms, available_video_keys) if include_videos else []
        )
        if include_videos and requested_video_keys:
            logging.info("Restricting local video loading to keys: %s", requested_video_keys)
        delta_timestamps = _build_delta_timestamps(
            data_config,
            fps,
            action_horizon,
            include_videos=include_videos,
        )
        def _build_one_local(root):
            return LocalLeRobotV3Dataset(
                root,
                delta_timestamps=delta_timestamps,
                tolerance_s=data_config.tolerance_s,
                video_backend=video_backend,
                include_videos=include_videos,
                video_keys=requested_video_keys if requested_video_keys else None,
                precomputed_video_root=data_config.precomputed_video_root,
            )

        if len(local_roots) <= 1:
            datasets_list = [_build_one_local(root) for root in local_roots]
        else:
            # Each repo eager-reads its data parquet at build time, which is bound by per-file
            # latency on network storage (tens of seconds for a repo of a few hundred parquet
            # files), so building many repos serially takes hours. The parquet read releases the
            # GIL, so a thread pool overlaps those reads and cuts it to minutes. Memory stays
            # eager and grows with the number of repos, so size the job's RAM accordingly.
            import concurrent.futures

            max_workers = min(16, len(local_roots))
            logging.info("Building %d local datasets with a %d-thread pool...", len(local_roots), max_workers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                datasets_list = list(ex.map(_build_one_local, local_roots))
        if data_config.repo_group_ids is not None:
            # Per-repo norm-group routing (norm_grouping="robot_action_schema"):
            # tag every sample with its repo's group id so GroupedNormalize can pick the right
            # per-group stats. repo_group_ids is ordered parallel to repo_ids/local_roots.
            if len(data_config.repo_group_ids) != len(datasets_list):
                raise ValueError(
                    f"repo_group_ids has {len(data_config.repo_group_ids)} entries for "
                    f"{len(datasets_list)} local datasets"
                )
            logging.info(
                "Tagging %d local datasets with norm_group_id (%d groups: %s)",
                len(datasets_list),
                len(set(data_config.repo_group_ids)),
                sorted(set(data_config.repo_group_ids)),
            )
            datasets_list = [
                TransformedDataset(ds, [_InjectConstants({"norm_group_id": gid})])
                for ds, gid in zip(datasets_list, data_config.repo_group_ids, strict=True)
            ]
        dataset = datasets_list[0] if len(datasets_list) == 1 else torch.utils.data.ConcatDataset(datasets_list)
    else:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id or repo_ids[0])
        task_mapping = dataset_meta.tasks
        delta_timestamps = _build_delta_timestamps(
            data_config,
            dataset_meta.fps,
            action_horizon,
            include_videos=include_videos,
        )
        if repo_ids:
            dataset = lerobot_dataset.MultiLeRobotDataset(
                list(repo_ids),
                delta_timestamps=delta_timestamps,
                tolerances_s={repo: data_config.tolerance_s for repo in repo_ids},
                video_backend=video_backend,
            )
        else:
            dataset = lerobot_dataset.LeRobotDataset(
                data_config.repo_id,
                delta_timestamps=delta_timestamps,
                tolerance_s=data_config.tolerance_s,
                video_backend=video_backend,
            )

    if not include_videos and not (local_roots and all(_is_local_lerobot_v3_root(root) for root in local_roots)):
        _disable_lerobot_video_loading(dataset)

    if data_config.exclude_anomalous_episodes:
        dataset_roots = repo_ids if repo_ids else (typing.cast(str, data_config.repo_id),)
        kept_indices = _build_non_anomalous_indices(dataset_roots)
        if kept_indices:
            dataset = IndexFilteredDataset(dataset, kept_indices)
        else:
            logging.warning("Anomaly filtering produced an empty index set; falling back to the unfiltered dataset.")

    if data_config.prompt_from_task:
        if repo_ids:
            raise NotImplementedError("prompt_from_task is not supported with multiple LeRobot dataset roots.")
        if task_mapping is None:
            raise RuntimeError("task mapping was not initialized for the LeRobot dataset")
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(task_mapping)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Compute them for this dataset with "
                "`python scripts/compute_canonical_norm.py --repo-id <dataset> --robot <flexiv|aloha> "
                "--train-config-name <config> --asset-id <asset-id>`, and make sure the asset id "
                "matches the one the config resolves (VTLA_ASSET_ID for the reference configs)."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. Compute them for this dataset with "
                "`python scripts/compute_canonical_norm.py --repo-id <dataset> --robot <flexiv|aloha> "
                "--train-config-name <config> --asset-id <asset-id>`, and make sure the asset id "
                "matches the one the config resolves (VTLA_ASSET_ID for the reference configs)."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with n0vtla.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
