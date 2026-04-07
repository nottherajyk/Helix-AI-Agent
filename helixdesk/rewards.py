import numpy as np
from dataclasses import dataclass
from typing import Optional
from helixdesk.simulator.email_gen import EmailEvent
from helixdesk.simulator.employee_sim import TickResolutionEvent

@dataclass
class RewardEvent:
    event_type: str
    value: float
    ticket_id: Optional[str]
    details: str

class RewardFunction:
    def __init__(self, config: dict):
        self.cfg = config.get("rewards", {})

    def compute(
        self,
        action: np.ndarray,
        email: EmailEvent,
        resolution_events: list[TickResolutionEvent],
        trend_alerts: list[str],
        kb_updated: bool,
        employee_loads: list[int],
        prev_employee_loads: list[int],
    ) -> tuple[float, list[RewardEvent]]:
        """
        Compute total reward. Return (total_reward, list_of_events).
        Total reward MUST be clipped to [-1.0, +1.0] before returning.
        """
        events = []
        
        # 1. Classification vs ground truth
        predicted_types = ["query", "complaint", "pending_review"]
        predicted_type = predicted_types[action[0]]
        
        if action[0] != 2: # if not flag_for_human_review
            if predicted_type == email.ticket_type:
                events.append(RewardEvent("correct_priority", float(self.cfg.get("correct_priority", 0.5)), email.email_id, ""))
            else:
                events.append(RewardEvent("misclassification", float(self.cfg.get("misclassification", -0.5)), email.email_id, ""))

        # 2. Keyword flag
        if email.has_keyword_flag and predicted_type != "complaint":
            events.append(RewardEvent("keyword_flag_missed", float(self.cfg.get("keyword_flag_missed", -0.3)), email.email_id, ""))

        # 3. Resolution outcomes
        for ev in resolution_events:
            if ev.resolved:
                events.append(RewardEvent("resolve_on_time", float(self.cfg.get("resolve_on_time", 1.0)), ev.ticket_id, ""))
                if ev.csat_score and ev.csat_score >= 4:
                    events.append(RewardEvent("csat_high", float(self.cfg.get("csat_high", 0.8)), ev.ticket_id, ""))
                elif ev.csat_score and ev.csat_score <= 2:
                    events.append(RewardEvent("bad_autoreply", float(self.cfg.get("bad_autoreply", -0.8)), ev.ticket_id, ""))
            else:
                events.append(RewardEvent("missed_deadline", float(self.cfg.get("missed_deadline", -1.0)), ev.ticket_id, ""))

        # 4. Trend alerts
        for cat in trend_alerts:
            events.append(RewardEvent("trend_prevented", float(self.cfg.get("trend_prevented", 0.6)), None, cat))

        # 5. Workload balance
        if len(employee_loads) > 1 and np.std(employee_loads) < np.std(prev_employee_loads):
            events.append(RewardEvent("balanced_assignment", float(self.cfg.get("balanced_assignment", 0.4)), None, ""))

        # 6. KB update
        if kb_updated:
            events.append(RewardEvent("kb_updated", float(self.cfg.get("kb_updated", 0.3)), None, ""))

        # 7. Unnecessary escalation
        if action[0] == 2:  # flag_for_human_review
            events.append(RewardEvent("unnecessary_escalation", float(self.cfg.get("unnecessary_escalation", -0.6)), email.email_id, ""))

        total = float(np.clip(sum(e.value for e in events), -1.0, 1.0))
        return total, events
