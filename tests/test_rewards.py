import numpy as np
from helixdesk.rewards import RewardFunction, RewardEvent
from helixdesk.simulator.email_gen import EmailEvent
from helixdesk.simulator.employee_sim import TickResolutionEvent

def get_base_cfg():
    return {
        "evaluate_rewards": True,
        "rewards": {
            "resolve_on_time": 1.0,
            "csat_high": 0.8,
            "trend_prevented": 0.6,
            "correct_priority": 0.5,
            "balanced_assignment": 0.4,
            "kb_updated": 0.3,
            "missed_deadline": -1.0,
            "bad_autoreply": -0.8,
            "unnecessary_escalation": -0.6,
            "misclassification": -0.5,
            "complaint_reopened": -0.4,
            "keyword_flag_missed": -0.3
        }
    }

def test_reward_clipping():
    rf = RewardFunction(get_base_cfg())
    action = np.array([1, 1, 0, 0]) # complaint
    email = EmailEvent("1", "a@a.com", "billing_dispute", "complaint", "text", 0.5, False, "free", "medium", 0.0)
    res = [TickResolutionEvent("1", True, 5), TickResolutionEvent("2", True, 5)] 
    r, evs = rf.compute(action, email, res, [], False, [1,1,1], [2,2,2])
    assert r == 1.0
    
def test_keyword_flag_missed():
    rf = RewardFunction(get_base_cfg())
    action = np.array([0, 1, 0, 0]) # predict query
    email = EmailEvent("1", "a@a.com", "billing_dispute", "complaint", "text", 0.5, True, "free", "critical", 0.0)
    r, evs = rf.compute(action, email, [], [], False, [], [])
    types = [e.event_type for e in evs]
    assert "keyword_flag_missed" in types
