import numpy as np
import gymnasium.spaces as spaces

def build_observation_space() -> spaces.Box:
    """
    State space: 42 dimensions
    Values are mostly normalized in range [-1.0, 1.0] or [0.0, 1.0].
    """
    return spaces.Box(low=-1.0, high=1.0, shape=(42,), dtype=np.float32)

def build_action_space() -> spaces.MultiDiscrete:
    """
    Action space: MultiDiscrete([3, 4, 6, 3])
    - Dim 0 (Classification): 0=query, 1=complaint, 2=flag_for_human_review
    - Dim 1 (Priority): 0=critical, 1=high, 2=medium, 3=normal
    - Dim 2 (Assignment): 0=emp0, 1=emp1, 2=emp2, 3=emp3, 4=emp4, 5=no_assignment
    - Dim 3 (Secondary action): 0=auto_reply_from_kb, 1=alert_gm, 2=no_secondary
    """
    return spaces.MultiDiscrete([3, 4, 6, 3])
