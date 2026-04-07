import numpy as np
from dataclasses import dataclass

@dataclass
class TicketAssignment:
    ticket_id: str
    sla_deadline_minutes: float
    assigned_time_minutes: float

@dataclass
class TickResolutionEvent:
    ticket_id: str
    resolved: bool
    csat_score: float | None

class EmployeeSimulator:
    def __init__(self, n_employees: int, max_load: int, base_resolve_rate: float, overload_penalty: float, ignore_probability: float, seed: int = 42):
        self.n_employees = n_employees
        self.max_load = max_load
        self.base_resolve_rate = base_resolve_rate
        self.overload_penalty = overload_penalty
        self.ignore_probability = ignore_probability
        self.rng = np.random.default_rng(seed)
        self.queues: list[list[TicketAssignment]] = []
        self.resolve_times: list[list[float]] = []
        self.reset(seed)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.queues = [[] for _ in range(self.n_employees)]
        self.resolve_times = [[] for _ in range(self.n_employees)]

    def assign(self, employee_idx: int, ticket_id: str, sla_deadline_minutes: float, current_time: float):
        if employee_idx < 0 or employee_idx >= self.n_employees:
            raise ValueError(f"Invalid employee_idx: {employee_idx}")
        if len(self.queues[employee_idx]) >= self.max_load:
            raise ValueError(f"Employee {employee_idx} is overloaded past max {self.max_load}")
            
        self.queues[employee_idx].append(TicketAssignment(
            ticket_id=ticket_id,
            sla_deadline_minutes=sla_deadline_minutes,
            assigned_time_minutes=current_time
        ))

    def get_loads(self) -> list[int]:
        return [len(q) for q in self.queues]

    def get_avg_resolve_times(self) -> list[float]:
        avgs = []
        for times in self.resolve_times:
            if not times:
                avgs.append(0.0)
            else:
                avgs.append(float(np.mean(times)))
        return avgs

    def tick(self, current_time_minutes: float) -> list[TickResolutionEvent]:
        events = []
        possible_csat = [3, 4, 4, 4, 5, 5]
        
        for e_idx in range(self.n_employees):
            queue = self.queues[e_idx]
            if not queue:
                continue
                
            resolve_rate = self.base_resolve_rate
            if len(queue) == self.max_load:
                resolve_rate = max(0.1, resolve_rate - self.overload_penalty)
                
            new_queue = []
            for ticket in queue:
                # Agent chance to ignore or resolve
                if self.rng.random() < self.ignore_probability:
                    new_queue.append(ticket)
                    continue
                    
                if self.rng.random() < resolve_rate:
                    # Ticket resolved
                    resolve_time = current_time_minutes - ticket.assigned_time_minutes
                    self.resolve_times[e_idx].append(resolve_time)
                    missed_sla = current_time_minutes > ticket.sla_deadline_minutes
                    
                    csat = None
                    if not missed_sla:
                        csat = float(self.rng.choice(possible_csat))
                        
                    events.append(TickResolutionEvent(
                        ticket_id=ticket.ticket_id,
                        resolved=True,
                        csat_score=csat
                    ))
                else:
                    new_queue.append(ticket)
                    
            self.queues[e_idx] = new_queue
            
        return events
