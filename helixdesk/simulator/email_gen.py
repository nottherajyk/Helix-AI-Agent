import uuid
import numpy as np
from dataclasses import dataclass

@dataclass
class EmailEvent:
    email_id: str
    sender_email: str
    category: str
    ticket_type: str
    body_text: str
    sentiment_intensity: float
    has_keyword_flag: bool
    customer_tier: str
    true_priority: str
    created_at_minutes: float

class EmailGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.rng = np.random.default_rng(config.get("seed", 42))
        self.email_count = 0
        self.categories = config.get("categories", [
            "login_failure", "billing_dispute", "refund_request", "product_defect",
            "shipping_delay", "account_locked", "data_privacy", "general_query"
        ])
        self.keywords = ["legal action", "consumer forum", "SEBI", "RBI",
                         "IRDAI", "social media post", "going viral", "fraud",
                         "scam", "court", "police complaint", "lawsuit"]

    def reset(self, seed: int | None = None):
        self.email_count = 0
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
    def _get_tier(self) -> str:
        r = self.rng.random()
        e_rate = self.config.get("enterprise_rate", 0.10)
        s_rate = self.config.get("standard_rate", 0.55)
        if r < e_rate:
            return "enterprise"
        elif r < e_rate + s_rate:
            return "standard"
        return "free"

    def _get_true_priority(self, tier: str, sentiment: float, has_kw: bool) -> str:
        if has_kw:
            return "critical"
        if sentiment > 0.85 or tier == "enterprise":
            return "high"
        return "medium"

    def next(self, sim_time: float) -> EmailEvent:
        cat = str(self.rng.choice(self.categories))
        
        is_query = self.rng.random() < self.config.get("query_ratio", 0.45)
        ticket_type = "query" if is_query else "complaint"
        
        has_kw = self.rng.random() < self.config.get("keyword_flag_rate", 0.05)
        
        if is_query:
            sentiment = self.rng.beta(2, 5)
        else:
            sentiment = self.rng.beta(4, 2)
            
        if has_kw:
            sentiment = max(0.85, sentiment)
            
        tier = self._get_tier()
        priority = self._get_true_priority(tier, sentiment, has_kw)
        
        body = f"[{ticket_type.upper()}] regarding {cat}."
        if has_kw:
            body += f" Mentioned: {self.rng.choice(self.keywords)}"
            
        self.email_count += 1
        return EmailEvent(
            email_id=f"email_{self.email_count}",
            sender_email=f"customer_{self.rng.integers(1000,9999)}@example.com",
            category=cat,
            ticket_type=ticket_type,
            body_text=body,
            sentiment_intensity=float(sentiment),
            has_keyword_flag=has_kw,
            customer_tier=tier,
            true_priority=priority,
            created_at_minutes=sim_time
        )
