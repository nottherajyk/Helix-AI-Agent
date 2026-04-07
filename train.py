import argparse
from helixdesk.env import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent
from helixdesk.monitor import EpisodeLogger, TerminalDashboard

def run(agent_type: str, n_episodes: int):
    env = HelixDeskEnv()
    logger = EpisodeLogger()
    dashboard = TerminalDashboard()

    if agent_type == 'rule':
        agent = RuleAgent(env.observation_space, env.action_space)
    elif agent_type == 'random':
        agent = RandomAgent(env.observation_space, env.action_space)
    elif agent_type == 'sb3':
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=n_episodes * 100)
        model.save("helixdesk_ppo")
        return

    episode_rewards = []
    
    # Context manager for the dashboard
    with dashboard.live():
        for ep in range(n_episodes):
            obs, info = env.reset()
            agent.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Assign action to info to log it later correctly
                info["action"] = action
                
                agent.learn(obs, action, reward, obs, terminated, info)
                logger.log(ep, info)
                dashboard.update(ep, info, episode_rewards)
                ep_reward += float(reward)
                done = terminated or truncated
            episode_rewards.append(ep_reward)

    print(f"\nFinal avg reward (last 50 eps): {sum(episode_rewards[-50:]) / min(len(episode_rewards), 50):.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='rule', choices=['rule', 'random', 'sb3'])
    parser.add_argument('--episodes', type=int, default=200)
    args = parser.parse_args()
    run(args.agent, args.episodes)
