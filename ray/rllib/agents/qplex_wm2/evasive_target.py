"""Evasive Target Agent — moves toward warehouse while avoiding cameras.

Used during training to make camera actions affect target movement,
giving WM2 (RSSM) meaningful action-conditioned dynamics to learn.

Movement = direction_to_warehouse + α × avoidance_from_visible_cameras + noise

During evaluation, switch back to GreedyTargetAgent (no avoidance).
"""

import numpy as np

import mate
from mate.constants import WAREHOUSES, NUM_WAREHOUSES


class EvasiveTargetAgent(mate.TargetAgentBase):
    """Target that moves toward warehouse while actively avoiding cameras.

    Args:
        seed: random seed
        noise_scale: noise magnitude (default 0.5, same as GreedyTargetAgent)
        avoidance_strength: how strongly to avoid cameras (default 0.5)
            0.0 = GreedyTargetAgent behavior (no avoidance)
            1.0 = strong avoidance (equal weight to warehouse-seeking and camera-avoiding)
        avoidance_range: normalized distance within which cameras trigger avoidance (default 0.5)
    """

    def __init__(self, seed=None, noise_scale=0.5, avoidance_strength=0.5, avoidance_range=0.5):
        super().__init__(seed=seed)

        self.noise_scale = float(noise_scale)
        self.avoidance_strength = float(avoidance_strength)
        self.avoidance_range = float(avoidance_range)

        self.goal_bits = None
        self.prev_state = None
        self.prev_noise = None
        self.non_empty_warehouses = set(range(NUM_WAREHOUSES))
        self.need_communication = False

    @property
    def goal(self):
        if self.goal_bits is not None and self.goal_bits.any():
            return np.flatnonzero(self.goal_bits)[0]
        return None

    @property
    def goal_location(self):
        goal = self.goal
        if goal is not None:
            return WAREHOUSES[goal]
        return None

    def reset(self, observation):
        super().reset(observation)
        self.prev_state = self.state
        self.prev_noise = 0.5 * self.action_space.sample()
        self.goal_bits = self.state.goal_bits.copy()
        self.non_empty_warehouses = set(range(NUM_WAREHOUSES))
        self.need_communication = False

    def observe(self, observation, info=None):
        self.state, observation, info, messages = self.check_inputs(observation, info)
        self.process_messages(observation, messages)

    def act(self, observation, info=None, deterministic=None):
        self.state, observation, info, _ = self.check_inputs(observation, info)

        # --- Goal selection (same as GreedyTargetAgent) ---
        if self.state.goal_bits.any():
            self.goal_bits = self.state.goal_bits
        if self.goal is None or (
            not self.state.goal_bits.any() and self.goal not in self.non_empty_warehouses
        ):
            self.goal_bits = np.zeros_like(self.state.goal_bits)
            if len(self.non_empty_warehouses) > 0:
                new_goal = self.np_random.choice(list(self.non_empty_warehouses))
                self.goal_bits[new_goal] = 1

        prev_actual_action = self.state.location - self.prev_state.location

        # --- Warehouse-seeking action ---
        if self.goal is not None:
            action = self.goal_location - self.state.location
        else:
            action = np.zeros_like(self.state.location)
        step_size = np.linalg.norm(action)
        if step_size > self.state.step_size:
            action *= self.state.step_size / step_size

        # --- Camera avoidance ---
        avoidance = self._compute_avoidance(observation)
        action = action + self.avoidance_strength * avoidance

        # Re-normalize to step_size
        action_norm = np.linalg.norm(action)
        if action_norm > self.state.step_size:
            action *= self.state.step_size / action_norm

        # --- Noise (same as GreedyTargetAgent) ---
        prob = 0.05 if np.linalg.norm(prev_actual_action) > 0.2 * self.state.step_size else 0.75
        if self.np_random.binomial(1, prob) != 0:
            noise = self.noise_scale * self.action_space.sample()
        else:
            noise = self.prev_noise

        action = (action + noise).clip(min=self.action_space.low, max=self.action_space.high)

        self.prev_state = self.state
        self.prev_noise = noise
        return action

    def _compute_avoidance(self, observation):
        """Compute avoidance vector: move away from visible cameras.

        Returns a vector pointing AWAY from the weighted average of nearby visible cameras.
        Closer cameras contribute more to the avoidance direction.
        """
        avoidance = np.zeros(2)

        # Get camera states from observation (opponents for target = cameras)
        try:
            camera_states, visible_flags = self.get_all_opponent_states(observation)
        except (ValueError, IndexError):
            return avoidance

        for camera_state, is_visible in zip(camera_states, visible_flags):
            if not is_visible:
                continue

            # Direction from camera to self (= away from camera)
            diff = self.state.location - camera_state.location
            dist = np.linalg.norm(diff)

            if dist < 1e-6 or dist > self.avoidance_range * 2000.0:
                continue

            # Inverse distance weighting: closer cameras → stronger avoidance
            weight = 1.0 / (dist + 1e-6)
            avoidance += weight * diff

        # Normalize to step_size
        avoidance_norm = np.linalg.norm(avoidance)
        if avoidance_norm > 1e-6:
            avoidance = avoidance / avoidance_norm * self.state.step_size

        return avoidance

    def process_messages(self, observation, messages):
        seen_empty_warehouses = set(np.flatnonzero(self.state.empty_bits))
        if len(seen_empty_warehouses.intersection(self.non_empty_warehouses)) > 0:
            self.non_empty_warehouses.difference_update(seen_empty_warehouses)
            self.need_communication = True

    def send_responses(self):
        messages = []
        if self.need_communication:
            content = {'non_empty_warehouses': self.non_empty_warehouses.copy()}
            messages.append(self.pack_message(content=content))
            self.need_communication = False
        return messages

    def receive_responses(self, messages):
        self.last_responses = tuple(messages)
        for message in self.last_responses:
            content = message.content
            if 'non_empty_warehouses' in content:
                self.non_empty_warehouses.intersection_update(content['non_empty_warehouses'])
