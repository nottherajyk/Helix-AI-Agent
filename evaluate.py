import argparse
import numpy as np
from rich.console import Console
from rich.table import Table
from helixdesk.env import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent

def evaluate(agent_type: str, n_episodes: int):
    env = HelixDeskEnv()
    if agent_type == 'rule':
        agent = RuleAgent(env.observation_space, env.action_space)
    elif agent_type == 'random':
        agent = RandomAgent(env.observation_space, env.action_space)
    else:
        print("Only rule and random supported for basic evaluation script.")
        return

    ep_rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        agent.reset()
        ep_r = 0.0
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += float(reward)
            done = terminated or truncated
        ep_rewards.append(ep_r)
        
    mean_reward = np.mean(ep_rewards)
    std_reward = np.std(ep_rewards)
    
    console = Console()
    table = Table(title=f"Evaluation Results ({n_episodes} episodes)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Agent", agent_type)
    table.add_row("Mean Reward", f"{mean_reward:.3f} +/- {std_reward:.3f}")
    
    # Note: additional metrics added as placeholders for full impl
    table.add_row("Mean Overdue Rate", "N/A (Simplified)")
    table.add_row("Mean CSAT", "N/A (Simplified)")
    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="rule", choices=["rule", "random"])
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    evaluate(args.agent, args.episodes)
