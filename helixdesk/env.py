import yaml
import numpy as np
import gymnasium as gym
from typing import Any, Tuple

from helixdesk.spaces import build_observation_space, build_action_space
from helixdesk.rewards import RewardFunction
from helixdesk.simulator.clock import SimClock
from helixdesk.simulator.email_gen import EmailGenerator
from helixdesk.simulator.employee_sim import EmployeeSimulator
from helixdesk.simulator.knowledge_base import KnowledgeBase
from helixdesk.simulator.trend_watchdog import TrendWatchdog

class HelixDeskEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode: str | None = None, config_path='config.yaml'):
        super().__init__()
        self.render_mode = render_mode
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        env_cfg = self.config.get("env", {})
        sla_cfg = self.config.get("sla", {})
        
        # Simulators
        self.clock = SimClock(seed=env_cfg.get("seed", 42))
        self.email_generator = EmailGenerator(self.config.get("email_gen", {}))
        self.employee_simulator = EmployeeSimulator(
            n_employees=env_cfg.get("n_employees", 5),
            max_load=sla_cfg.get("max_employee_load", 10),
            base_resolve_rate=self.config.get("employee_sim", {}).get("base_resolve_rate", 0.7),
            overload_penalty=self.config.get("employee_sim", {}).get("overload_penalty", 0.3),
            ignore_probability=self.config.get("employee_sim", {}).get("ignore_probability", 0.05),
            seed=env_cfg.get("seed", 42)
        )
        self.knowledge_base = KnowledgeBase(self.config.get("email_gen", {}).get("categories", []))
        self.trend_watchdog = TrendWatchdog(
            window_hours=env_cfg.get("trend_window_hours", 72.0),
            threshold_pct=env_cfg.get("trend_alert_threshold", 30.0)
        )
        
        # Reward Function
        self.reward_function = RewardFunction(self.config)

        # Spaces
        self.observation_space = build_observation_space()
        self.action_space = build_action_space()

        # Ep parameters
        self.episode_emails = env_cfg.get("episode_emails", 100)
        
        # Internal state
        self._queue = []
        self._current_email = None
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_info = {}
        
        self.spec = gym.envs.registration.EnvSpec(
            id=self.config.get("openenv", {}).get("env_id", "HelixDesk-v1"),
            entry_point="helixdesk.env:HelixDeskEnv"
        )
        
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.clock.reset(seed)
        self.employee_simulator.reset(seed)
        self.trend_watchdog.reset()
        self.knowledge_base.reset()
        self.email_generator.reset(seed)
        
        self._queue = []
        self._step_count = 0
        self._episode_reward = 0.0
        
        self._current_email = self.email_generator.next(self.clock.minutes)
        
        # SLA info tracking
        self._queue.append(self._current_email)
        
        obs = self.state()
        info = {
            "step": self._step_count,
            "email_id": self._current_email.email_id,
            "sim_time_minutes": self.clock.minutes,
            "ticket_type": self._current_email.ticket_type,
            "priority": "normal",
            "assigned_to": None,
            "reward_breakdown": {},
            "queue_depth": 1,
            "overdue_count": 0,
            "trend_alerts_active": 0,
            "csat_score": None,
            "episode_reward_so_far": 0.0
        }
        self._episode_info = info
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Decode action
        predicted_types = ["query", "complaint", "pending_review"]
        classification = predicted_types[action[0]]
        
        if action[0] == 2: # flag_for_human_review forces (3, 5, 2)
            action[1], action[2], action[3] = 3, 5, 2

        priorities = ["critical", "high", "medium", "normal"]
        priority = priorities[action[1]]
        
        assigned_to = action[2] if action[2] != 5 else None
        
        secondary_action = action[3]
        kb_updated = False
        
        if secondary_action == 0 and classification == "query":
            # auto_reply
            pass
        elif secondary_action == 1:
            # alert_gm
            pass

        # Apply action
        sla_hours = self.config.get("sla", {})
        sla_map = {
            "critical": sla_hours.get("critical_hours", 4),
            "high": sla_hours.get("high_hours", 12),
            "medium": sla_hours.get("medium_hours", 24),
            "normal": sla_hours.get("normal_hours", 48)
        }
        deadline = self.clock.minutes + sla_map[priority] * 60.0
        
        if assigned_to is not None:
            # attempt to assign
            try:
                self.employee_simulator.assign(assigned_to, self._current_email.email_id, deadline, self.clock.minutes)
            except ValueError:
                pass # skip assignment if overloaded

        # Step time
        self.clock.tick()
        
        # Resolutions
        resolution_events = self.employee_simulator.tick(self.clock.minutes)
        resolved_ids = {e.ticket_id for e in resolution_events if e.resolved}
        self._queue = [e for e in self._queue if e.email_id not in resolved_ids]
        
        # Trend
        self.trend_watchdog.record(self._current_email.category, self.clock.minutes)
        alerts = self.trend_watchdog.tick(self.clock.minutes)
        
        # Reward
        prev_loads = self.employee_simulator.get_loads()
        curr_loads = prev_loads # they are same here since assign happens before tick, but simplified
        
        reward, evs = self.reward_function.compute(
            action, self._current_email, resolution_events, alerts, kb_updated, curr_loads, prev_loads
        )
        self._episode_reward += reward
        self._step_count += 1
        
        # Next email
        self._current_email = self.email_generator.next(self.clock.minutes)
        self._queue.append(self._current_email)
        
        # State
        obs = self.state()
        
        terminated = bool(self._step_count >= self.episode_emails)
        truncated = False
        
        # Build info
        self._episode_info = {
            "step": self._step_count,
            "sim_time_minutes": self.clock.minutes,
            "email_id": self._current_email.email_id,
            "ticket_type": classification,
            "priority": priority,
            "assigned_to": assigned_to,
            "reward_breakdown": {e.event_type: e.value for e in evs},
            "queue_depth": len(self._queue),
            "overdue_count": 0, # simplified for now
            "trend_alerts_active": len(alerts),
            "csat_score": sum(e.csat_score for e in resolution_events if e.csat_score) / max(1, sum(1 for e in resolution_events if e.csat_score)),
            "episode_reward_so_far": self._episode_reward
        }
        
        return obs, float(reward), terminated, truncated, self._episode_info

    def _build_obs_vector(self) -> np.ndarray:
        obs = np.zeros(42, dtype=np.float32)
        
        email = self._current_email
        obs[0] = float(email.sentiment_intensity)
        obs[1] = 1.0 if email.has_keyword_flag else 0.0
        obs[2] = 1.0 if email.customer_tier == "enterprise" else 0.0
        obs[3] = 1.0 if email.customer_tier == "standard" else 0.0
        obs[4] = 1.0 if email.customer_tier == "free" else 0.0
        
        # Category one-hot
        cats = self.config.get("email_gen", {}).get("categories", [])
        idx = cats.index(email.category) if email.category in cats else 0
        if idx < 5:
            obs[5 + idx] = 1.0
        else:
            obs[9] = float(idx) / len(cats) # overflow
            
        loads = self.employee_simulator.get_loads()
        for i, l in enumerate(loads):
            obs[15 + i*2] = min(1.0, l / max(1, self.config.get("sla", {}).get("max_employee_load", 10)))
            
        # Sim Time
        obs[37] = (self.clock.minutes / 60.0 % 24) / 24.0
        obs[38] = (self.clock.minutes / (60.0 * 24) % 7) / 7.0
        
        obs[39] = max(0.0, (self.episode_emails - self._step_count) / self.episode_emails)
        obs[40] = np.clip(self._episode_reward / self.episode_emails, -1.0, 1.0)
        obs[41] = 1.0 # agent_confidence
        
        return obs

    def state(self) -> np.ndarray:
        return self._build_obs_vector()
        
    def render(self, mode='human'):
        s = f"Step: {self._step_count} Reward: {self._episode_reward:.2f}"
        if mode == 'human':
            print(s)
        return s
        
    def close(self):
        pass
