"""QuadX Waypoints Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class QuadXWaypointsEnv(QuadXBaseEnv):
    """QuadX Waypoints Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is a set of `[x, y, z, (optional) yaw]` waypoints in space.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        num_targets (int): number of waypoints in the environment.
        use_yaw_targets (bool): whether to match yaw targets before a waypoint is considered reached.
        goal_reach_distance (float): distance to the waypoints for it to be considered reached.
        goal_reach_angle (float): angle in radians to the waypoints for it to be considered reached, only in effect if `use_yaw_targets` is used.
        flight_mode (int): the flight mode of the UAV.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution.

    """

    def __init__(
        self,
        sparse_reward: bool = False,
        num_targets: int = 4,
        use_yaw_targets: bool = False,
        goal_reach_distance: float = 0.2,
        goal_reach_angle: float = 0.1,
        flight_mode: int = 0,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            num_targets (int): number of waypoints in the environment.
            use_yaw_targets (bool): whether to match yaw targets before a waypoint is considered reached.
            goal_reach_distance (float): distance to the waypoints for it to be considered reached.
            goal_reach_angle (float): angle in radians to the waypoints for it to be considered reached, only in effect if `use_yaw_targets` is used.
            flight_mode (int): the flight mode of the UAV.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution.

        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # define waypoints
        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            min_height=0.1,
            np_random=self.np_random,
        )

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "target_deltas": spaces.Sequence(
                    space=spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(4,) if use_yaw_targets else (3,),
                        dtype=np.float64,
                    ),
                    stack=True,
                ),
            }
        )

        """ ENVIRONMENT CONSTANTS """
        self.sparse_reward = sparse_reward

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict[Literal["attitude", "target_deltas"], np.ndarray], dict]:
        """Resets the environment.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(seed, options)
        self.waypoints.reset(self.env, self.np_random)
        self.info["num_targets_reached"] = 0
        super().end_reset()

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep.

        This returns the observation as well as the distances to target.
        - "attitude" (Box)
        ----- ang_vel (vector of 3 values)
        ----- ang_pos (vector of 3/4 values)
        ----- lin_vel (vector of 3 values)
        ----- lin_pos (vector of 3 values)
        ----- previous_action (vector of 4 values)
        ----- auxiliary information (vector of 4 values)
        - "target_deltas" (Sequence)
        ----- list of body_frame distances to target (vector of 3/4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # combine everything
        new_state: dict[Literal["attitude", "target_deltas"], np.ndarray] = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    ang_pos,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                ],
                axis=-1,
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.concatenate(
                [
                    ang_vel,
                    quaternion,
                    lin_vel,
                    lin_pos,
                    self.action,
                    aux_state,
                ],
                axis=-1,
            )

        new_state["target_deltas"] = self.waypoints.distance_to_targets(
            ang_pos, lin_pos, quaternion
        )

        self.state: dict[Literal["attitude", "target_deltas"], np.ndarray] = new_state

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward()

        # bonus reward if we are not sparse
        if not self.sparse_reward:
            self.reward += max(3.0 * self.waypoints.progress_to_next_target, 0.0)
            self.reward += 0.1 / self.waypoints.distance_to_next_target
            # Negative Reward For High Yaw rate, To prevent high yaw while training
            yaw_rate = abs(
                self.env.state(0)[0][2]
            )  # Assuming z-axis is the last component
            yaw_rate_penalty = 0.01 * yaw_rate**2  # Add penalty for high yaw rate
            self.reward -= (
                yaw_rate_penalty  # You can adjust the coefficient (0.01) as needed
            )

        # target reached
        if self.waypoints.target_reached:
            self.reward = 100.0

            # advance the targets
            self.waypoints.advance_targets()

            # update infos and dones
            self.truncation |= self.waypoints.all_targets_reached
            self.info["env_complete"] = self.waypoints.all_targets_reached
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached
