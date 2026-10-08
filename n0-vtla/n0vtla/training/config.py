"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Mapping, Sequence
import dataclasses
import difflib
import json
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import n0vtla.models.model as _model
import n0vtla.models.pi0_config as pi0_config
import n0vtla.models.tokenizer as _tokenizer
import n0vtla.policies.canonical_tactile_policy as canonical_tactile_policy
import n0vtla.policies.canonical_schema as cs
import n0vtla.policies.flexiv_policy as flexiv_policy
import n0vtla.shared.download as _download
import n0vtla.shared.normalize as _normalize
import n0vtla.training.droid_rlds_dataset as droid_rlds_dataset
import n0vtla.training.optimizer as _optimizer
import n0vtla.training.weight_loaders as weight_loaders
import n0vtla.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

_FLEXIV_TACTILE_KEYS = (
    "observation.image.left_wrist_left_tactile",
    "observation.image.left_wrist_right_tactile",
)
# Tactile action-predictor (v0): a large negative delta_timestamps offset (in frames) that
# LeRobot clamps to the episode's FIRST frame, used to load the tactile baseline. Must exceed
# the longest episode length. Ported from the reference data_loader._LATENT_BASELINE_FRAMES.
_LATENT_BASELINE_FRAMES = 100_000


def _short_tactile_name(raw_key: str) -> str:
    """`observation.image.left_wrist_left_tactile` -> `left_wrist_left_tactile` (the model-side
    view name used by FlexivTactileInputs and N0VTLAConfig.tactile_image_keys)."""
    return raw_key.rsplit(".", 1)[-1]


def _get_local_dataset_fps(repo_id: str | None, default_fps: float = 30.0) -> float:
    if repo_id is None:
        return default_fps

    info_path = pathlib.Path(repo_id) / "meta" / "info.json"
    if not info_path.is_file():
        return default_fps

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return float(info["fps"])
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        logging.warning("Falling back to default tactile-flow fps=%.1f for %s", default_fps, repo_id)
        return default_fps


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional list of LeRobot dataset roots that should be concatenated for training.
    repo_ids: tuple[str, ...] | None = None
    # Optional per-repo norm-group ids (ordered parallel to repo_ids). When set, the data loader
    # tags every sample with its repo's ``norm_group_id`` so GroupedNormalize can route to the
    # right per-group stats (norm_grouping="robot_action_schema").
    repo_group_ids: tuple[str, ...] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()
    # If true, local LeRobot datasets will drop episodes marked as anomalous in metadata.json.
    exclude_anomalous_episodes: bool = False
    # Optional reduced transform pipeline used by norm-stat computation. This can skip image loading for
    # datasets where the statistics only depend on state/actions.
    stats_transforms: Sequence[_transforms.DataTransformFn] | None = None
    # Optional per-key timestamp offsets (in seconds) that should also be queried by the dataset loader.
    extra_delta_timestamps: Mapping[str, Sequence[float]] | None = None
    # Optional root directory that contains precomputed video-frame caches for local LeRobot v3 datasets.
    # Files are resolved as `<root>/<dataset_dir_name>/<video_key>.npy`.
    precomputed_video_root: str | None = None
    # Timestamp tolerance in seconds when matching videos to frame timestamps.
    tolerance_s: float = 1e-4


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Optional list of LeRobot dataset roots that should be concatenated for training.
    repo_ids: tuple[str, ...] | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # If true, local LeRobot datasets will drop episodes marked as anomalous in metadata.json.
    exclude_anomalous_episodes: bool = False
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id
        if asset_id is None and repo_id is not None:
            asset_id = repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            repo_ids=self.repo_ids,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
            exclude_anomalous_episodes=self.exclude_anomalous_episodes,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LeRobotFlexivVisionDataConfig(DataConfigFactory):
    """Vision-only control: the EEF pipeline with no tactile pathway, on 32-dim canonical data.

    Mirrors LeRobotFlexivEEFDataConfig but adapts the two keys the canonical fruit dataset
    differs on:
    - state repacks from ``observation.state`` (32-dim canonical) instead of
      ``observation.state.eef_pose`` (this dataset has no ``.eef_pose`` sub-key). Feeding the
      full 32-dim state matches what the N0 parity run's CanonicalVTLAInputs feeds the base policy
      (the base policy max_action_dim is 32, so no synthetic zero-padding artifact).
    - the prompt comes from the LeRobot task (set ``base_config=DataConfig(prompt_from_task=True)``
      in the TrainConfig); an identity ``prompt`` entry keeps it because RepackTransform drops
      any key not present in its structure.

    Delta scheme is n0vtla-native element-wise delta: first 9 eef dims -> delta vs current
    state, gripper absolute (make_bool_mask(9, -1)). Padding dims 10..31 are constant zero and
    normalize to a constant (the +1e-6 guard in Normalize avoids divide-by-zero / NaN).
    """

    use_delta_eef_actions: bool = True
    default_prompt: str | None = None
    tolerance_s: float = 0.04
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.OptionalRepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.image.third_view",
                            "cam_left_wrist": "observation.image.left_wrist_view",
                            "cam_right_wrist": "observation.image.second_third_view",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    },
                    optional_source_keys=frozenset({"observation.image.second_third_view"}),
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[flexiv_policy.FlexivEEFInputs()],
            outputs=[flexiv_policy.FlexivEEFOutputs()],
        )
        if self.use_delta_eef_actions:
            delta_action_mask = _transforms.make_bool_mask(9, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
            stats_transforms = [
                _transforms.RepackTransform(
                    {
                        "state": "observation.state",
                        "actions": "action",
                    }
                ),
                _transforms.DeltaActions(delta_action_mask),
            ]
        else:
            stats_transforms = [
                _transforms.RepackTransform(
                    {
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            stats_transforms=stats_transforms,
            tolerance_s=self.tolerance_s,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotFlexivTactileDataConfig(DataConfigFactory):
    """Tactile action-predictor (v0, UNSUPERVISED) data config: clone of
    LeRobotFlexivVisionDataConfig (same 32-dim canonical state, n0vtla-native
    element-wise EEF delta) PLUS latent tactile [baseline(frame0), current].

    Difference from control:
      * repack adds a ``tactile`` sub-dict mapping each canonical tactile view name to its raw
        LeRobot column (``observation.image.<view>``);
      * ``FlexivTactileInputs`` (not ``FlexivEEFInputs``) splits each tactile column's
        delta_timestamps stack into ``<view>`` / ``<view>.baseline`` model image keys — which
        ``N0VTLAPolicy`` pops before SigLIP;
      * ``extra_delta_timestamps`` loads a 2-frame ``[baseline(frame0), current]`` stack per
        tactile column: offsets ``[-BIG, 0]`` frames (LeRobot clamps -BIG to frame 0). Ported
        from the reference b2/latent ``_build_delta_ts`` (data_loader.py:34-51), minus the
        future (t+H) frame (v0 is unsupervised).

    NOTE: the EEF-delta / stats_transforms block below is copied verbatim from
    LeRobotFlexivVisionDataConfig - keep the two in sync; only the ``tactile`` repack,
    ``FlexivTactileInputs`` input, and ``extra_delta_timestamps`` below differ.

    ``tactile_keys`` are the RAW LeRobot column names (delta_timestamps are keyed by raw column,
    applied before transforms). The model-side ``tactile_image_keys`` are the short view names.
    """

    use_delta_eef_actions: bool = True
    default_prompt: str | None = None
    tolerance_s: float = 0.04
    tactile_keys: Sequence[str] = _FLEXIV_TACTILE_KEYS
    action_sequence_keys: Sequence[str] = ("action",)
    future_frame_offset: int = 0        # >0: load tac_{t+offset} as the stage-1 z* target frame

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Map each raw tactile column into a `tactile` sub-dict that FlexivTactileInputs reads.
        tactile_repack = {
            _short_tactile_name(k): k for k in self.tactile_keys
        }
        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.OptionalRepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.image.third_view",
                            "cam_left_wrist": "observation.image.left_wrist_view",
                            "cam_right_wrist": "observation.image.second_third_view",
                        },
                        "tactile": tactile_repack,
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    },
                    optional_source_keys=frozenset({"observation.image.second_third_view"}),
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[flexiv_policy.FlexivTactileInputs()],
            outputs=[flexiv_policy.FlexivEEFOutputs()],
        )
        if self.use_delta_eef_actions:
            delta_action_mask = _transforms.make_bool_mask(9, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
            stats_transforms = [
                _transforms.RepackTransform(
                    {
                        "state": "observation.state",
                        "actions": "action",
                    }
                ),
                _transforms.DeltaActions(delta_action_mask),
            ]
        else:
            stats_transforms = [
                _transforms.RepackTransform(
                    {
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # v0: 2-frame [baseline(frame0), current] stack per tactile column via delta_timestamps.
        # Stage-1: future_frame_offset>0 appends tac_{t+offset} -> [baseline, current, future].
        fps = _get_local_dataset_fps(self.repo_id)
        tac_offsets = [-_LATENT_BASELINE_FRAMES / fps, 0.0]
        if self.future_frame_offset > 0:
            tac_offsets = tac_offsets + [self.future_frame_offset / fps]
        extra_delta_timestamps = {key: list(tac_offsets) for key in self.tactile_keys}

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            stats_transforms=stats_transforms,
            tolerance_s=self.tolerance_s,
            extra_delta_timestamps=extra_delta_timestamps,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotCanonicalTaskTactileDataConfig(DataConfigFactory):
    """Single-task canonical data pipeline for tactile-predictor post-training.

    The on-disk state and action use the canonical padded 32-dimensional EEF layout. Sensor
    slots are fixed across robot embodiments: missing RGB or tactile views are represented by
    zero-valued placeholders with false image masks. Actions are converted from absolute EEF
    poses to element-wise deltas while grippers and padding remain absolute.
    """

    default_prompt: str | None = None
    tolerance_s: float = 0.04
    use_delta_eef_actions: bool = True
    action_sequence_keys: Sequence[str] = ("action",)
    # >0: also load tac_{t+offset} as the Stage-1 z*/Dbar target frame (paper Sec 4.2; see
    # n0vtla_policy.py::N0VTLAPolicy._build_future_target and
    # docs/MID_TRAIN.md). 0 (default) -> 2-frame [baseline, current] stack. Set to the action
    # horizon H (50) to match the paper's target definition.
    future_frame_offset: int = 0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # self.repo_ids (inherited from DataConfigFactory, plumbed through
        # create_base_config -> DataConfig.repo_ids, which the data loader concatenates)
        # is normally None, in which case the single repo_id is used. Set repo_ids explicitly
        # to train on multiple LeRobot dataset roots at once.
        repo_ids = self.repo_ids if self.repo_ids else (self.repo_id,)

        stage1 = bool(getattr(model_config, "stage1_pretrain_enabled", False))
        if stage1 and self.future_frame_offset <= 0:
            raise ValueError("Stage 1 requires a positive future_frame_offset")

        repack_map: dict[str, str] = {key: key for key in cs.IMAGE_KEYS}
        repack_map["observation.state"] = "observation.state"
        repack_map["action"] = "action"
        repack_map["action_mask"] = "action_mask"
        for key in cs.TACTILE_KEYS:
            repack_map[key + "_is_pad"] = key + "_is_pad"

        repack_transforms = _transforms.Group(
            inputs=[canonical_tactile_policy.OptionalRepack(repack_map)]
        )
        data_transforms = _transforms.Group(
            inputs=[canonical_tactile_policy.CanonicalTactileInputs(flip_wrist_180=False)],
        )
        stats_transforms: list[_transforms.DataTransformFn] = [
            _transforms.RepackTransform({"state": "observation.state", "actions": "action"})
        ]
        if stage1:
            for key in ("observation.state", "action", "action_mask"):
                repack_map.pop(key, None)
            data_transforms = _transforms.Group(inputs=[
                canonical_tactile_policy.Stage1ObservationOnly(
                    model_config.action_dim, model_config.action_horizon
                ),
                *data_transforms.inputs,
            ])
            stats_transforms = []
        elif self.use_delta_eef_actions:
            # ChunkDeltaToCurrentState/RelRotAbsoluteActions rather than DeltaActions/
            # AbsoluteActions: the latter pair does plain element-wise subtraction/addition on
            # the whole masked eef block, including the 6 rot6d dims. rot6d is not a vector
            # space, so that is not a valid relative rotation. ChunkDeltaToCurrentState composes
            # rotation matrices instead (R_action @ R_state^T) and RelRotAbsoluteActions is its
            # exact inverse; xyz stays element-wise, which is valid for position.
            data_transforms = data_transforms.push(
                inputs=[_transforms.ChunkDeltaToCurrentState()],
                outputs=[_transforms.RelRotAbsoluteActions()],
            )
            stats_transforms.append(_transforms.ChunkDeltaToCurrentState())
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        fps = _get_local_dataset_fps(self.repo_id)
        tactile_offsets = [-_LATENT_BASELINE_FRAMES / fps, 0.0]
        if self.future_frame_offset > 0:
            tactile_offsets = tactile_offsets + [self.future_frame_offset / fps]
        extra_delta_timestamps = {
            key: list(tactile_offsets) for key in cs.TACTILE_KEYS
        }

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repo_ids=repo_ids,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=() if stage1 else self.action_sequence_keys,
            stats_transforms=stats_transforms,
            tolerance_s=self.tolerance_s,
            extra_delta_timestamps=extra_delta_timestamps,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotSimTactileJointDataConfig(DataConfigFactory):
    """Simulation joint-space data config for the latent tactile predictor.

    This is the loader behind the two released simulation checkpoints. It matches the schema the
    UniVTAC and NeoSim simulators write: three RGB columns (``top`` / ``wrist_l`` / ``wrist_r``) plus one
    video column per tactile sensor, joint-space actions, and a canonical top-level
    ``observation.state`` / ``action``.

    Tactile is routed through the *latent* path, not the direct-injection ``expert_images`` path:
    ``extra_delta_timestamps`` loads every tactile column as ``[baseline(frame 0), current]`` and
    ``FlexivTactileInputs`` splits that stack into ``<view>`` / ``<view>.baseline`` model image
    keys, which ``N0VTLAPolicy._preprocess_observation`` pops back out before SigLIP.

    Set ``tactile_keys`` / ``raw_action_dim`` to match the embodiment:

      * single arm — 2 sensors, ``raw_action_dim=8``  (7 joints + gripper)
      * dual arm   — 4 sensors, ``raw_action_dim=16`` (2 x (7 joints + gripper))

    ``use_delta_joint_actions`` predicts joint deltas but leaves each gripper column absolute,
    which is what the released checkpoints were trained with. The mask is derived from
    ``raw_action_dim`` so the two embodiments stay in sync.
    """

    use_delta_joint_actions: bool = True
    raw_action_dim: int = 8
    default_prompt: str | None = None
    tolerance_s: float = 0.04
    tactile_keys: Sequence[str] = (
        "observation.images.tactile_a",
        "observation.images.tactile_b",
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The right-wrist camera is optional: single-arm simulator datasets have no
        # ``observation.images.wrist_r`` column at all. A plain RepackTransform would raise
        # KeyError on the first batch; the optional variant drops the slot instead, and
        # FlexivTactileInputs then fills it with a zero image and image_mask=False, which is
        # exactly how the single-arm checkpoint was trained.
        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.OptionalRepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.top",
                            "cam_left_wrist": "observation.images.wrist_l",
                            "cam_right_wrist": "observation.images.wrist_r",
                        },
                        "tactile": {
                            _short_tactile_name(key): key for key in self.tactile_keys
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    },
                    optional_source_keys=frozenset({"observation.images.wrist_r"}),
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[flexiv_policy.FlexivTactileInputs()],
            outputs=[flexiv_policy.FlexivJointOutputs(action_dim=self.raw_action_dim)],
        )

        stats_repack = _transforms.RepackTransform(
            {"state": "observation.state", "actions": "action"}
        )
        if self.use_delta_joint_actions:
            # Per arm: 7 joint dims are deltas, the gripper column stays absolute.
            arms = max(1, self.raw_action_dim // 8)
            delta_action_mask = _transforms.make_bool_mask(*((7, -1) * arms))
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
            stats_transforms = [stats_repack, _transforms.DeltaActions(delta_action_mask)]
        else:
            stats_transforms = [stats_repack]

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        fps = _get_local_dataset_fps(self.repo_id)
        extra_delta_timestamps = {
            key: [-_LATENT_BASELINE_FRAMES / fps, 0.0] for key in self.tactile_keys
        }

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            stats_transforms=stats_transforms,
            tolerance_s=self.tolerance_s,
            extra_delta_timestamps=extra_delta_timestamps,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "n0vtla"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    #
    # Inference DROID configs.
    #
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        # Vision-only parity baseline: vanilla PI0Pytorch with no tactile pathway, on the same
        # reference data and with the same lr / steps / batch size / base weights / precision as
        # the tactile run, but using element-wise EEF deltas (use_delta_eef_actions) rather than
        # ChunkDeltaToCurrentState. Useful for deciding whether a training artefact comes from
        # the data or from the tactile modifications.
        name="flexiv_vision_reference",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50, pytorch_compile_mode=None),
        data=LeRobotFlexivVisionDataConfig(
            repo_id=os.environ.get("VTLA_DATASET_PATH", "/path/to/datasets/flexiv_reference_task"),
            use_delta_eef_actions=True,
            default_prompt="do the task",
            assets=AssetsConfig(
                asset_id="flexiv_vision_reference",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        num_workers=8,
        log_interval=50,
        save_interval=5_000,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path=os.environ.get("VTLA_PRETRAINED_CHECKPOINT"),
        num_train_steps=20_000,
        wandb_enabled=False,
    ),
    TrainConfig(
        # Tactile action-predictor (v0, UNSUPERVISED): clone of flexiv_vision_reference (SAME
        # data root, element-wise EEF delta, constant 5e-5 / 20k / bs64) + the config-gated
        # predictor path. Run via `scripts/train_n0vtla.py` (monkey-patches PI0Pytorch
        # -> N0VTLAPolicy). tactile_predictor_enabled=True activates the g->z predictor prepended to the
        # action expert; loss is action_mse only (Predictor trained by the action gradient). No z*,
        # no future frames. use_tactile stays False so the base direct-injection modules aren't built.
        name="flexiv_tactile_reference",
        # __import__ indirection keeps config.py's top-level import graph free of the optional
        # tactile-predictor module (N0VTLAPolicy, torch-heavy); only this flag-gated entry needs it.
        model=(
            lambda: __import__(
                "n0vtla.models_pytorch.n0vtla_policy", fromlist=["N0VTLAConfig"]
            ).N0VTLAConfig(
                pi05=True,
                action_dim=32,
                action_horizon=50,
                pytorch_compile_mode=None,
                tactile_predictor_enabled=True,
                n_latent=5,
                tactile_image_keys=("left_wrist_left_tactile", "left_wrist_right_tactile"),
            )
        )(),
        data=LeRobotFlexivTactileDataConfig(
            repo_id=os.environ.get("VTLA_DATASET_PATH", "/path/to/datasets/flexiv_reference_task"),
            use_delta_eef_actions=True,
            default_prompt="do the task",
            tactile_keys=_FLEXIV_TACTILE_KEYS,
            assets=AssetsConfig(
                asset_id="flexiv_vision_reference",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        num_workers=8,
        log_interval=50,
        save_interval=5_000,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path=os.environ.get("VTLA_PRETRAINED_CHECKPOINT"),
        num_train_steps=20_000,
        wandb_enabled=False,
    ),
    # Reference configuration for tactile-conditioned post-training.
    TrainConfig(
        name="vtla_tactile_posttrain",
        model=(
            lambda: __import__(
                "n0vtla.models_pytorch.n0vtla_policy", fromlist=["N0VTLAConfig"]
            ).N0VTLAConfig(
                pi05=True,
                action_dim=32,
                action_horizon=50,
                pytorch_compile_mode=None,
                tactile_predictor_enabled=True,
                tactile_mode="latent",
                n_latent=5,
                predictor_arch="tactile_kv",
                z_gate_zero_init=True,
                vl_dropout_prob=0.0,
                predictor_loss_weight=0.0,
                tactile_image_keys=cs.TACTILE_SHORT_KEYS,
            )
        )(),
        data=LeRobotCanonicalTaskTactileDataConfig(
            repo_id=os.environ.get("VTLA_DATASET_PATH", "/path/to/datasets/canonical_tactile_task"),
            tolerance_s=0.04,
            use_delta_eef_actions=True,
            default_prompt=os.environ.get("VTLA_DEFAULT_PROMPT", "Perform the task."),
            assets=AssetsConfig(
                asset_id=os.environ.get("VTLA_ASSET_ID", "canonical_tactile_task"),
            ),
        ),
        batch_size=64,
        num_workers=8,
        log_interval=100,
        save_interval=5_000,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=2e-5,
            decay_steps=20_000,
            decay_lr=2e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path=os.environ.get(
            "VTLA_PRETRAINED_CHECKPOINT", "/path/to/checkpoints/vtla_pretrained"
        ),
        num_train_steps=20_000,
        wandb_enabled=True,
    ),
    # Stage-1 predictor-grounding pretraining (paper Sec 4.2) -- action-FREE: trains only
    # tactile_encoder.tactile_proj + tactile_predictor + tactile_recon_head against an InfoNCE +
    # L1-recon future-tactile target, never touches ground-truth actions. See
    # docs/MID_TRAIN.md for the design and scripts/train_stage1_predictor.py for the (separate,
    # action-free) training loop that consumes this config. action_dim/action_horizon are
    # inherited from Pi0Config only because N0VTLAPolicy subclasses PI0Pytorch; the Stage-1 loss
    # never constructs the action suffix.
    TrainConfig(
        name="vtla_stage1_predictor_pretrain",
        model=(
            lambda: __import__(
                "n0vtla.models_pytorch.n0vtla_policy", fromlist=["N0VTLAConfig"]
            ).N0VTLAConfig(
                pi05=True,
                action_dim=32,
                action_horizon=50,
                pytorch_compile_mode=None,
                tactile_predictor_enabled=True,
                tactile_mode="latent",
                # n_latent=5 (not the paper's 10): the released n0-vtla-base checkpoint's
                # TactileActionPredictor.latent_queries was built with n_latent=5, so changing
                # it would shape-mismatch that parameter and prevent warm-starting from that
                # checkpoint. The InfoNCE math (Eq. 3-4) mean-pools z and z* independently
                # before the cosine-similarity matrix, so it is well-defined regardless of
                # whether the two sides share a token count. z*'s own token count (10, from
                # FrozenDINOv2TactileEncoder's 1+pool_grid**2 layout) is unaffected and matches
                # the paper's encoder architecture.
                n_latent=5,
                predictor_arch="tactile_kv",
                stage1_pretrain_enabled=True,
                stage1_recon_grid=int(os.environ.get("VTLA_STAGE1_RECON_GRID", "8")),
                stage1_lambda_rec=float(os.environ.get("VTLA_STAGE1_LAMBDA_REC", "0.5")),
                # Eq. 3-4's logits are unscaled cosine similarity (temperature=1); the default
                # matches that. See N0VTLAConfig.stage1_temperature.
                stage1_temperature=float(os.environ.get("VTLA_STAGE1_TEMPERATURE", "1.0")),
                tactile_image_keys=cs.TACTILE_SHORT_KEYS,
            )
        )(),
        data=LeRobotCanonicalTaskTactileDataConfig(
            repo_id=os.environ.get("VTLA_DATASET_PATH", "/path/to/datasets/canonical_tactile_task"),
            tolerance_s=0.04,
            use_delta_eef_actions=True,
            default_prompt=os.environ.get("VTLA_DEFAULT_PROMPT", "Perform the task."),
            # H=50 matches the paper's z* horizon (Eq. 2) and the model's action_horizon above;
            # the two are independent config surfaces that happen to share the same paper symbol.
            future_frame_offset=int(os.environ.get("VTLA_STAGE1_FUTURE_OFFSET", "50")),
            assets=AssetsConfig(
                asset_id=os.environ.get("VTLA_ASSET_ID", "canonical_tactile_task"),
            ),
        ),
        batch_size=64,
        num_workers=8,
        log_interval=50,
        # Checkpoint every 1000 steps: this bounds the work redone on --resume to <1000 steps
        # while keeping checkpoint I/O modest, since each trainable-only checkpoint is only
        # ~1-1.5GB (see save_stage1_checkpoint in scripts/train_stage1_predictor.py). Kept as a
        # config default (like lr_schedule below) so every launch picks it up without needing a
        # per-job override.
        save_interval=1000,
        keep_period=10_000,
        # peak_lr/decay_lr are 10x lower than the paper's from-scratch-grounding scale
        # (1e-4 -> 1e-5) because this config warm-starts from n0-vtla-base, which already
        # carries an action-aligned tactile predictor (the released checkpoint includes the
        # tactile encoder, latent tactile predictor, and projection parameters). This is
        # continued pretraining on an already-trained module rather than grounding from random
        # init, so the learning rate is kept small to perturb it as little as possible while
        # still letting human-data exposure shift the grounding.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-5,
            decay_steps=20_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path=os.environ.get(
            "VTLA_PRETRAINED_CHECKPOINT", "/path/to/checkpoints/vtla_pretrained"
        ),
        num_train_steps=20_000,
        wandb_enabled=False,
    ),
    # Fine-tuning DROID configs.
    #
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    #
    # Debugging configs.
    #
    # ---------------------------------------------------------------- released sim checkpoints
    # Configs for the two released simulation checkpoints, both post-trained from the N0-VTLA
    # pretrained base: sim_single_arm_tactile on UniVTAC, sim_dual_arm_tactile on NeoSim.
    #
    #   sim_single_arm_tactile -> single-arm tasks, 2 tactile views, 8-dim joint actions
    #   sim_dual_arm_tactile   -> dual-arm tasks,   4 tactile views, 16-dim joint actions
    #
    # Note the action space: these predict JOINT actions (7 joints + gripper per arm), whereas
    # the pretrained base and vtla_tactile_posttrain predict end-effector deltas in the canonical
    # 32-dim rot6d container. Do not reuse an end-effector normalisation asset with these.
    #
    # Serving needs no dataset — scripts/serve_zmq.py reads only model.safetensors plus
    # assets/<asset_id>/norm_stats.json. Point VTLA_DATASET_PATH and VTLA_ASSET_ID at your own
    # data only when you retrain.
    TrainConfig(
        name="sim_single_arm_tactile",
        model=(
            lambda: __import__(
                "n0vtla.models_pytorch.n0vtla_policy", fromlist=["N0VTLAConfig"]
            ).N0VTLAConfig(
                pi05=True,
                action_dim=32,
                action_horizon=50,
                pytorch_compile_mode=None,
                tactile_predictor_enabled=True,
                tactile_mode="latent",
                n_latent=5,
                predictor_arch="tactile_kv",
                # The released sim checkpoints carry NO z_gate parameter: they were trained with
                # the latent tokens entering the action expert ungated. Setting this True here
                # would CREATE a zero-initialised gate that the checkpoint cannot fill, silently
                # multiplying the whole tactile pathway by 0 at inference.
                z_gate_zero_init=False,
                vl_dropout_prob=0.0,
                predictor_loss_weight=0.0,
                tactile_image_keys=("tactile_a", "tactile_b"),
            )
        )(),
        data=LeRobotSimTactileJointDataConfig(
            repo_id=os.environ.get("VTLA_DATASET_PATH", "/path/to/datasets/sim_single_arm"),
            raw_action_dim=8,
            tactile_keys=(
                "observation.images.tactile_a",
                "observation.images.tactile_b",
            ),
            default_prompt=os.environ.get("VTLA_DEFAULT_PROMPT", "do the task"),
            assets=AssetsConfig(
                asset_id=os.environ.get("VTLA_ASSET_ID", "sim_single_arm_norm"),
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        num_workers=8,
        log_interval=50,
        save_interval=2_000,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path=os.environ.get("VTLA_PRETRAINED_CHECKPOINT"),
        num_train_steps=20_000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="sim_dual_arm_tactile",
        model=(
            lambda: __import__(
                "n0vtla.models_pytorch.n0vtla_policy", fromlist=["N0VTLAConfig"]
            ).N0VTLAConfig(
                pi05=True,
                action_dim=32,
                action_horizon=16,
                pytorch_compile_mode=None,
                tactile_predictor_enabled=True,
                tactile_mode="latent",
                n_latent=5,
                predictor_arch="tactile_kv",
                # See the note on sim_single_arm_tactile: these checkpoints have no z_gate.
                z_gate_zero_init=False,
                vl_dropout_prob=0.0,
                predictor_loss_weight=0.0,
                tactile_image_keys=("tactile_a", "tactile_b", "tactile_c", "tactile_d"),
            )
        )(),
        data=LeRobotSimTactileJointDataConfig(
            repo_id=os.environ.get("VTLA_DATASET_PATH", "/path/to/datasets/sim_dual_arm"),
            raw_action_dim=16,
            tactile_keys=(
                "observation.images.tactile_a",
                "observation.images.tactile_b",
                "observation.images.tactile_c",
                "observation.images.tactile_d",
            ),
            default_prompt=os.environ.get("VTLA_DEFAULT_PROMPT", "do the task"),
            assets=AssetsConfig(
                asset_id=os.environ.get("VTLA_ASSET_ID", "sim_dual_arm_norm"),
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=64,
        num_workers=8,
        log_interval=50,
        save_interval=2_000,
        keep_period=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path=os.environ.get("VTLA_PRETRAINED_CHECKPOINT"),
        num_train_steps=20_000,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
