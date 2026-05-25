import os
import json
import numpy as np
import torch
import zarr
from typing import Optional
from CleanDiffuser.cleandiffuser.utils import MinMaxNormalizer, create_indices
from CleanDiffuser.image_codecs import jpeg  # noqa


class RealmanDataset(torch.utils.data.Dataset):
    """Dataset for Realman dual-camera Zarr produced by scripts/collect.py.

    obs:
        rgb_arm  (To, 3, H, W) float32 in [0, 1]
        rgb_fix  (To, 3, H, W) float32 in [0, 1]
        pos      (To, 7)       float32 - normalized 6-dim EEF pose + 1-dim gripper_state
    action:
        (Ta, 7) float32 - normalized 6-dim spacemouse delta pose + 1-dim gripper_action
    """

    def __init__(
        self,
        file_path,
        To: int = 1,
        Ta: int = 64,
        normalizer_path: Optional[str] = None,
    ):
        super().__init__()
        self.root = zarr.open(file_path, "r")

        self.To, self.Ta = To, Ta
        self._episode_ends = self.root.meta.episode_ends[:]

        self.indices = create_indices(
            episode_ends=self._episode_ends,
            sequence_length=Ta + To - 1,
            pad_before=To - 1,
            pad_after=Ta - 1,
        )
        self.episode_idx = np.empty((len(self.indices),), dtype=int)
        for i in range(len(self.indices)):
            end_idx = self.indices[i][-1]
            self.episode_idx[i] = np.searchsorted(self._episode_ends, end_idx)

        self.size = self.root.data.pos.shape[0]

        self._init_normalizers(normalizer_path)

        self._obs_meta = [
            "rgb_arm",
            "rgb_fix",
            "pos",
            "gripper_state",
        ]

    def _init_normalizers(self, normalizer_path: Optional[str]):
        """Initialize MinMax normalizers for pos and action from JSON.

        Expected JSON schema (produced by scripts/validate.py):
        {
          "pos":    {"max": [..6..], "min": [..6..]},
          "action": {"max": [..6..], "min": [..6..]}
        }
        gripper_state and gripper_action are already in [0, 1] and are not
        normalized further.
        """
        if not (normalizer_path and os.path.exists(normalizer_path)):
            raise ValueError(f"Normalizer file {normalizer_path} does not exist")

        with open(normalizer_path, "r") as f:
            info = json.load(f)

        pos_max = np.array(info["pos"]["max"], dtype=np.float32)
        pos_min = np.array(info["pos"]["min"], dtype=np.float32)
        act_max = np.array(info["action"]["max"], dtype=np.float32)
        act_min = np.array(info["action"]["min"], dtype=np.float32)

        assert pos_max.shape[0] == 6 and pos_min.shape[0] == 6, "pos normalizer must be 6-dim"
        assert act_max.shape[0] == 6 and act_min.shape[0] == 6, "action normalizer must be 6-dim"

        self.pos_normalizer = MinMaxNormalizer(X_max=pos_max, X_min=pos_min)
        self.action_normalizer = MinMaxNormalizer(X_max=act_max, X_min=act_min)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            end_idx,
        ) = self.indices[idx]

        e_Ta = self.Ta - (self.To + self.Ta - 1 - sample_end_idx)

        action_6d = self.root.data.action[buffer_end_idx - e_Ta : buffer_end_idx]
        action_6d = self.action_normalizer.normalize(action_6d)

        gripper_action = self.root.data.gripper_action[buffer_end_idx - e_Ta : buffer_end_idx]

        # 6-dim delta pose + 1-dim gripper_action = 7-dim action
        action = np.concatenate((action_6d, gripper_action), axis=-1)
        if self.To + self.Ta - 1 > sample_end_idx:
            action = np.pad(
                action,
                ((0, self.To + self.Ta - 1 - sample_end_idx), (0, 0)),
                mode="edge",
            )
        assert action.shape[0] == self.Ta, f"{action.shape[0]} != {self.Ta}"

        observation = dict()
        for obs_name in self._obs_meta:
            this_start_idx = buffer_start_idx
            x = self.root.data[obs_name][
                this_start_idx : buffer_start_idx + self.To - sample_start_idx
            ]
            if sample_start_idx > 0:
                if obs_name == "rgb_arm" or obs_name == "rgb_fix":
                    x = np.pad(x, ((sample_start_idx, 0), (0, 0), (0, 0), (0, 0)), mode="edge")
                else:
                    x = np.pad(x, ((sample_start_idx, 0), (0, 0)), mode="edge")

            if obs_name == "pos":
                x = self.pos_normalizer.normalize(x).astype(np.float32)
            elif obs_name == "rgb_arm" or obs_name == "rgb_fix":
                x = x.astype(np.float32) / 255.0
            elif obs_name == "gripper_state":
                x = x.astype(np.float32)  # already in [0, 1]
            observation[obs_name] = x

        pos_with_gripper = np.concatenate(
            (observation["pos"], observation["gripper_state"]), axis=-1
        )

        return {
            "obs": {
                "pos": torch.tensor(pos_with_gripper, dtype=torch.float32),  # (To, 7)
                "rgb_arm": torch.tensor(observation["rgb_arm"], dtype=torch.float32),  # (To, C, H, W)
                "rgb_fix": torch.tensor(observation["rgb_fix"], dtype=torch.float32),  # (To, C, H, W)
            },
            "action": torch.tensor(action.astype(np.float32), dtype=torch.float32),  # (Ta, 7)
        }
