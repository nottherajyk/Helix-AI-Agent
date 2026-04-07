import numpy as np
from helixdesk.env import HelixDeskEnv

def test_reset_returns_correct_shape():
    env = HelixDeskEnv('config.yaml')
    obs, info = env.reset()
    assert obs.shape == (42,)
    assert obs.dtype == np.float32
    assert "step" in info

def test_step_returns_correct_structure():
    env = HelixDeskEnv('config.yaml')
    env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (42,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "reward_breakdown" in info

def test_episode_terminates_at_correct_step():
    env = HelixDeskEnv('config.yaml')
    env.reset()
    for _ in range(99):
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert not terminated
    _, _, terminated, _, _ = env.step(env.action_space.sample())
    assert terminated

def test_state_is_idempotent():
    env = HelixDeskEnv('config.yaml')
    env.reset()
    obs1 = env.state()
    obs2 = env.state()
    assert np.array_equal(obs1, obs2)

def test_reset_resets_sim_clock():
    env = HelixDeskEnv('config.yaml')
    env.reset()
    for _ in range(20):
        env.step(env.action_space.sample())
    obs1, _ = env.reset()
    obs2, _ = env.reset()
    assert np.array_equal(obs1[37], obs2[37])
