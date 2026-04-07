import numpy as np
from .base_agent import AbstractAgent

class RuleAgent(AbstractAgent):
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Action dims:
        0: Classification (0=query, 1=complaint, 2=flag_for_human_review)
        1: Priority (0=critical, 1=high, 2=medium, 3=normal)
        2: Assignment (0=emp0, ..., 4=emp4, 5=no_assignment)
        3: Secondary action (0=auto_reply_from_kb, 1=alert_gm, 2=no_secondary)
        """
        # obs dims:
        # 1: has_keyword_flag
        # 2: enterprise_tier
        # 0: sentiment
        # 15-24: employee loads
        
        has_kw = observation[1] > 0.5
        tier_ent = observation[2] > 0.5
        sentiment = observation[0]
        
        # Parse loads
        loads = [observation[15 + i*2] for i in range(5)]
        best_emp = int(np.argmin(loads))
        all_max = all(l > 0.99 for l in loads)
        
        if has_kw:
            return np.array([1, 0, best_emp, 2]) # complaint, critical, best_emp, no_sec
        if sentiment > 0.85:
            return np.array([1, 1, best_emp, 2]) # complaint, high
        if tier_ent:
            return np.array([1, 1, best_emp, 2])
        if all_max:
            return np.array([2, 3, 5, 2]) # flag for review -> forces (3, 5, 2)
            
        # check query pattern (simplification based on sentiment < 0.5)
        if sentiment < 0.5:
            return np.array([0, 3, best_emp, 0]) # query, auto_reply
            
        return np.array([1, 2, best_emp, 2]) # complaint, medium
