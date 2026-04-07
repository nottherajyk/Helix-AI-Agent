import pytest
from gymnasium.utils.env_checker import check_env
from helixdesk.env import HelixDeskEnv

def test_gymnasium_compatibility():
    env = HelixDeskEnv('config.yaml')
    check_env(env.unwrapped if hasattr(env, 'unwrapped') else env, warn=True)
