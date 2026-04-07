import pytest
from helixdesk.simulator.clock import SimClock
from helixdesk.simulator.email_gen import EmailGenerator
from helixdesk.simulator.employee_sim import EmployeeSimulator
from helixdesk.simulator.knowledge_base import KnowledgeBase
from helixdesk.simulator.trend_watchdog import TrendWatchdog
import numpy as np

def test_sim_clock():
    clock = SimClock()
    clock.tick()
    assert clock.minutes >= 8.0

def test_email_gen():
    gen = EmailGenerator({"categories": ["billing"]})
    email = gen.next(0.0)
    assert email.category == "billing"
    assert email.email_id is not None

def test_employee_sim_assign():
    sim = EmployeeSimulator(1, 1, 1.0, 0.0, 0.0)
    sim.assign(0, "A", 10.0, 0.0)
    with pytest.raises(ValueError):
        sim.assign(0, "B", 10.0, 0.0)

def test_trend_watchdog():
    watch = TrendWatchdog(1.0, 10.0)
    for _ in range(10):
        watch.record("billing", 10.0)
    alerts = watch.tick(10.0)
    assert isinstance(alerts, list)
    
def test_knowledge_base():
    kb = KnowledgeBase(["billing"])
    e, sim = kb.lookup("billing", 1.0)
    assert e is not None
    assert sim == 1.0
