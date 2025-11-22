import os
import gym
import numpy as np
import pybullet_envs           # register Bullet environments
from sac_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    train_sac = True      # Set to False to skip SAC training
    n_games   = 500       # Number of episodes to train / evaluate, was 250 in the other 2 envs
    env_id    = 'HopperBulletEnv-v0'   # ← choose the environment'
    plot_dir  = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

   
    # Training SAC Agent
    
    if train_sac:
        env = gym.make(env_id)
        agent = Agent(input_dims=env.observation_space.shape,
                      env=env,
                      n_actions=env.action_space.shape[0])

        score_history = []
        best_score    = env.reward_range[0]
        load_checkpoint = False

        for i in range(n_games):
            obs, done, score = env.reset(), False, 0
            while not done:
                action = agent.choose_action(obs)
                obs_, reward, done, _ = env.step(action)
                score += reward
                agent.remember(obs, action, reward, obs_, done)
                if not load_checkpoint:
                    agent.learn()
                obs = obs_

            score_history.append(score)
            avg100 = np.mean(score_history[-100:])
            if avg100 > best_score:
                best_score = avg100
                if not load_checkpoint:
                    agent.save_models()
            print(f"SAC | Episode {i+1} | Score: {score:.1f} | Avg: {avg100:.1f}")

        # save SAC learning curve
        sac_plot = os.path.join(plot_dir, f"{env_id.replace('/', '_')}_sac.png")
        plot_learning_curve(range(1, n_games+1), score_history, sac_plot)
        print(f"SAC learning curve saved to {sac_plot}")

   
    # Random baseline comparison
    

    env = gym.make(env_id)
    random_scores = []

    for i in range(n_games):
        obs, done, score = env.reset(), False, 0
        while not done:
            action = env.action_space.sample()
            obs, r, done, _ = env.step(action)
            score += r
        random_scores.append(score)

        if (i+1) % 50 == 0:
            avg100 = np.mean(random_scores[-100:])
            print(f"Random | Episode {i+1} | Avg: {avg100:.1f}")

    random_plot = os.path.join(plot_dir, f"{env_id.replace('/', '_')}_random.png")
    plot_learning_curve(range(1, n_games+1), random_scores, random_plot)
    print(f"Random baseline curve saved to {random_plot}")

    env.close()
