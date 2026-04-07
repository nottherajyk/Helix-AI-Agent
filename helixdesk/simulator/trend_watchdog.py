from collections import deque
from dataclasses import dataclass

@dataclass
class TrendRecord:
    category: str
    sim_time: float

class TrendWatchdog:
    def __init__(self, window_hours: float = 72.0, threshold_pct: float = 30.0):
        self.window_minutes = window_hours * 60.0
        self.threshold_pct = threshold_pct
        self.history: deque[TrendRecord] = deque()

    def reset(self):
        self.history.clear()

    def record(self, category: str, sim_time: float):
        self.history.append(TrendRecord(category, sim_time))

    def _get_counts(self, current_time: float) -> tuple[dict[str, int], dict[str, int]]:
        # current window: [current_time - window_minutes, current_time]
        # prior window: [current_time - 2 * window_minutes, current_time - window_minutes)
        current_counts = {}
        prior_counts = {}
        
        while self.history and self.history[0].sim_time < current_time - 2 * self.window_minutes:
            self.history.popleft()

        for rec in self.history:
            if rec.sim_time >= current_time - self.window_minutes:
                current_counts[rec.category] = current_counts.get(rec.category, 0) + 1
            else:
                prior_counts[rec.category] = prior_counts.get(rec.category, 0) + 1
                
        return current_counts, prior_counts

    def get_growth_rates(self, current_time: float) -> dict[str, float]:
        current_counts, prior_counts = self._get_counts(current_time)
        metrics = {}
        
        cats = set(current_counts.keys()).union(set(prior_counts.keys()))
        for cat in cats:
            curr = current_counts.get(cat, 0)
            prior = prior_counts.get(cat, 0)
            
            # growth = (current_window_count - prior_window_count) / max(prior, 1) * 100
            growth = ((curr - prior) / max(prior, 1)) * 100.0
            metrics[cat] = growth
            
        return metrics

    def tick(self, current_time: float) -> list[str]:
        """Returns list of categories that trigger a surge alert."""
        growth_rates = self.get_growth_rates(current_time)
        alerts = []
        for cat, growth in growth_rates.items():
            if growth >= self.threshold_pct:
                alerts.append(cat)
        return alerts
