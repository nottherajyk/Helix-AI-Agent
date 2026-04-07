import csv
import os

class EpisodeLogger:
    def __init__(self, log_path='./logs/', filename='episodes.csv'):
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        self.filepath = os.path.join(log_path, filename)
        
        write_header = not os.path.exists(self.filepath)
        self.file = open(self.filepath, 'a', newline='')
        self.writer = csv.writer(self.file)
        if write_header:
            self.writer.writerow([
                "episode", "step", "sim_time", "action_classify", "action_priority", "action_assign",
                "action_secondary", "reward", "episode_reward", "queue_depth", "overdue_count",
                "trend_alerts", "ticket_type", "priority_assigned"
            ])

    def log(self, ep: int, info: dict):
        self.writer.writerow([
            ep,
            info.get("step", 0),
            info.get("sim_time_minutes", 0),
            info.get("action", [0,0,0,0])[0] if "action" in info else "",
            info.get("action", [0,0,0,0])[1] if "action" in info else "",
            info.get("action", [0,0,0,0])[2] if "action" in info else "",
            info.get("action", [0,0,0,0])[3] if "action" in info else "",
            str(info.get("reward_breakdown", {})),
            info.get("episode_reward_so_far", 0),
            info.get("queue_depth", 0),
            info.get("overdue_count", 0),
            info.get("trend_alerts_active", 0),
            info.get("ticket_type", ""),
            info.get("priority", "")
        ])
        # Force flush to ensure it's written during training
        self.file.flush()

    def close(self):
        self.file.close()
