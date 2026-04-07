import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Any
from gymnasium.utils.env_checker import check_env
from helixdesk.env import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent

@dataclass
class GradeResult:
    check_name: str
    passed: bool
    score: float          # 0.0 – 1.0
    detail: str

def run_all_graders(n_episodes: int = 20) -> dict[str, Any]:
    results = []
    results += check_api_compliance()
    results += check_reward_signal(n_episodes)
    results += check_episode_correctness(n_episodes)
    results += check_agent_benchmark(n_episodes)
    results += check_safeguards()

    total_score = float(sum(r.score for r in results) / len(results))
    return {
        "total_score": round(total_score, 4),
        "checks": [asdict(r) for r in results],
        "env_id": "HelixDesk-v1",
        "passed_all": bool(all(r.passed for r in results)),
    }

def check_api_compliance() -> list[GradeResult]:
    results = []
    env = HelixDeskEnv()

    # Check 1: gymnasium env_checker
    try:
        check_env(env.unwrapped if hasattr(env, 'unwrapped') else env, warn=True)
        results.append(GradeResult("gymnasium_env_checker", True, 1.0, "check_env passed with 0 errors"))
    except Exception as e:
        results.append(GradeResult("gymnasium_env_checker", False, 0.0, str(e)))

    # Check 2: observation shape on reset
    obs, info = env.reset()
    shape_ok = obs.shape == (42,) and obs.dtype == np.float32
    results.append(GradeResult("observation_shape", bool(shape_ok), float(shape_ok), f"Got shape={obs.shape} dtype={obs.dtype}"))

    # Check 3: info dict has required keys
    required_keys = {"step", "sim_time_minutes", "email_id", "ticket_type",
                     "priority", "reward_breakdown", "queue_depth",
                     "overdue_count", "episode_reward_so_far"}
    _, _, _, _, step_info = env.step(env.action_space.sample())
    missing = required_keys - set(step_info.keys())
    keys_ok = len(missing) == 0
    results.append(GradeResult("info_dict_keys", bool(keys_ok), float(keys_ok), f"Missing keys: {missing}" if missing else "All keys present"))

    # Check 4: state() is idempotent
    s1, s2 = env.state(), env.state()
    idem_ok = np.array_equal(s1, s2)
    results.append(GradeResult("state_idempotent", bool(idem_ok), float(idem_ok), "state() is idempotent" if idem_ok else "state() changed between calls"))

    env.close()
    return results

def check_reward_signal(n_episodes: int) -> list[GradeResult]:
    results = []
    env = HelixDeskEnv()
    rewards = []

    for _ in range(n_episodes):
        env.reset()
        ep_rewards = []
        done = False
        while not done:
            _, r, terminated, truncated, _ = env.step(env.action_space.sample())
            ep_rewards.append(r)
            done = terminated or truncated
        rewards.extend(ep_rewards)

    env.close()

    clipped_ok = all(-1.0 <= r <= 1.0 for r in rewards)
    results.append(GradeResult("reward_clipped", bool(clipped_ok), float(clipped_ok), f"Min={min(rewards) if rewards else 0:.3f} Max={max(rewards) if rewards else 0:.3f}"))

    diversity_ok = len(set(round(r, 3) for r in rewards)) > 5
    results.append(GradeResult("reward_diversity", bool(diversity_ok), float(diversity_ok), f"Unique reward values: {len(set(round(r,3) for r in rewards))}"))

    return results

def check_episode_correctness(n_episodes: int) -> list[GradeResult]:
    results = []
    env = HelixDeskEnv()
    correct_terminates = 0

    for _ in range(n_episodes):
        env.reset()
        steps = 0
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            done = terminated or truncated
        if steps == 100:
            correct_terminates += 1

    env.close()
    rate = correct_terminates / n_episodes
    results.append(GradeResult("episode_length_correct", bool(rate == 1.0), float(rate), f"{correct_terminates}/{n_episodes} episodes terminated at step 100"))
    return results

def check_agent_benchmark(n_episodes: int) -> list[GradeResult]:
    results = []

    def run_agent(AgentClass):
        env = HelixDeskEnv()
        obs, _ = env.reset()
        agent = AgentClass(env.observation_space, env.action_space)
        ep_rewards = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            agent.reset()
            ep_r = 0.0
            done = False
            while not done:
                action = agent.act(obs)
                obs, r, terminated, truncated, info = env.step(action)
                agent.learn(obs, action, r, obs, terminated, info)
                ep_r += r
                done = terminated or truncated
            ep_rewards.append(ep_r)
        env.close()
        return np.mean(ep_rewards)

    rule_score = run_agent(RuleAgent)
    random_score = run_agent(RandomAgent)

    rule_beats_random = bool(rule_score > random_score)
    results.append(GradeResult("rule_beats_random", rule_beats_random, float(rule_beats_random), f"Rule={rule_score:.3f} Random={random_score:.3f}"))
    return results

def check_safeguards() -> list[GradeResult]:
    results = []
    env = HelixDeskEnv()
    
    overload_prevented = True
    try:
        from helixdesk.simulator.employee_sim import EmployeeSimulator
        sim = EmployeeSimulator(1, 1, 1.0, 0.0, 0.0)
        sim.assign(0, "e1", 10.0, 0.0)
        sim.assign(0, "e2", 10.0, 0.0)
        overload_prevented = False
    except ValueError:
        pass
        
    results.append(GradeResult("safeguard_overload_prevented", overload_prevented, 1.0 if overload_prevented else 0.0, "Overload prevented" if overload_prevented else "Overload allowed"))
    
    has_watchdog = hasattr(env, "trend_watchdog")
    results.append(GradeResult("safeguard_trend_watchdog", has_watchdog, 1.0 if has_watchdog else 0.0, "Watchdog present" if has_watchdog else "Watchdog missing"))
    
    return results
