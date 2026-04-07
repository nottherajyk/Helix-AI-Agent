from rich.live import Live
from rich.table import Table

class TerminalDashboard:
    def __init__(self):
        self.live_display = None

    def live(self):
        self.live_display = Live(self._build_table(0, {}, []), refresh_per_second=2)
        return self.live_display

    def update(self, ep: int, info: dict, episode_rewards: list):
        if not self.live_display: return
        self.live_display.update(self._build_table(ep, info, episode_rewards))

    def _build_table(self, ep: int, info: dict, episode_rewards: list) -> Table:
        table = Table(title="┌─── HelixDesk OpenEnv ───┐")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        step = info.get("step", 0)
        sim_time = info.get("sim_time_minutes", 0.0)
        ep_reward = info.get("episode_reward_so_far", 0.0)
        
        avg = sum(episode_rewards[-10:]) / len(episode_rewards[-10:]) if episode_rewards else 0.0
        
        table.add_row("Episode", str(ep))
        table.add_row("Step", f"{step}/100")
        table.add_row("Sim time", f"{int(sim_time//60)}h {int(sim_time%60)}m")
        table.add_row("Ep Reward", f"{ep_reward:+.2f}")
        table.add_row("Avg(last 10)", f"{avg:+.2f}")
        table.add_row("Queue Depth", str(info.get("queue_depth", 0)))
        
        return table
