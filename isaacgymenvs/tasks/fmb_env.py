"""FMBEnv: FabricaEnv subclass for FMB multi-object board assembly tasks.

Identical to FabricaEnv except:
- Assets loaded from assets/urdf/fmb/ instead of assets/urdf/fabrica/.
- Start pose Z comes from scenes.npz + 5cm (not tableObjectZOffset).

Usage (config):
    task=FMBBoardEnv task.env.assemblyName=fmb_board_1
"""

import torch
from isaacgymenvs.tasks.fabrica_env import FabricaEnv
from isaacgymenvs.utils.torch_jit_utils import torch_rand_float

START_Z_OFFSET = 0.05


class FMBEnv(FabricaEnv):
    ASSETS_SUBDIR = "fmb"

    def reset_object_pose(self, env_ids, reset_buf_idxs=None, tensor_reset=True):
        if tensor_reset and len(env_ids) > 0 and reset_buf_idxs is None:
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]

            force_start = int(self.cfg["env"].get("forceStartIdx", -1))
            if force_start >= 0:
                new_start = torch.full(
                    (len(env_ids),), force_start, dtype=torch.long, device=self.device)
            else:
                new_start = torch.randint(
                    0, self._si_m_starts, (len(env_ids),), device=self.device)
            self._si_env_start_idx_t[env_ids] = new_start

            part_ids = self._si_env_part_idx_t[env_ids]
            scene_ids = self._si_env_scene_idx_t[env_ids]

            self.env_max_goals[env_ids] = self._si_traj_lengths_t[part_ids, scene_ids, new_start]

            poses = self._si_start_poses_t[part_ids, scene_ids, new_start]
            self.object_init_state[env_ids, 0:3] = poses[:, 0:3]
            self.object_init_state[env_ids, 2] += START_Z_OFFSET
            self.object_init_state[env_ids, 3:7] = poses[:, 3:7]

            self.goal_pos_obs_noise[env_ids, 0:2] = torch_rand_float(
                -self.cfg["env"]["goalXyObsNoise"],
                self.cfg["env"]["goalXyObsNoise"],
                (len(env_ids), 2), device=self.device,
            )

        # Skip FabricaEnv.reset_object_pose, call SimToolReal directly.
        # SimToolReal.reset_object_pose will use object_init_state (which now
        # has XYZ from scenes.npz + offset) but will overwrite Z with
        # tableObjectZOffset. We need to prevent that by calling the grandparent.
        from isaacgymenvs.tasks.simtoolreal.env import SimToolReal
        SimToolReal.reset_object_pose(self, env_ids, reset_buf_idxs, tensor_reset)

        # Re-apply our Z after SimToolReal overwrote it.
        if tensor_reset and len(env_ids) > 0 and reset_buf_idxs is None:
            part_ids = self._si_env_part_idx_t[env_ids]
            scene_ids = self._si_env_scene_idx_t[env_ids]
            start_ids = self._si_env_start_idx_t[env_ids]
            poses = self._si_start_poses_t[part_ids, scene_ids, start_ids]

            obj_indices = self.object_indices[env_ids].to(torch.long)
            self.root_state_tensor[obj_indices, 2] = poses[:, 2] + START_Z_OFFSET
