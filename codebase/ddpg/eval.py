"""
evaluation.py

This module provides a function to evaluate the DDPG agent on the Unity environment.
During evaluation, the agent acts without exploration noise, and the environment is run
in evaluation mode (train_mode=False). The function maintains a sliding window of the last
100 evaluation episodes, and if the average reward over the window reaches a specified threshold,
it signals that training should stop.
"""

import numpy as np
from collections import deque

# Global sliding window to store rewards from the last 100 evaluation episodes.
evaluation_rewards_window = deque(maxlen=100)

def evaluate(agent, env, normalizer=None, episodes=5, threshold=30.0):
    """
    Evaluate the agent over a number of episodes and update a sliding window of recent rewards.
    
    Args:
        agent: The DDPGAgent instance with its learned policy.
        env: An instance of EnvWrapper for the Unity environment.
        normalizer (optional): A running normalizer to preprocess state observations.
        episodes (int): Number of evaluation episodes to run in this call.
        threshold (float): The target average reward over 100 episodes that defines success.
    
    Returns:
        avg_reward (float): The current sliding window average reward.
        solved (bool): True if the sliding window is full and its average reward >= threshold.
    """
    global evaluation_rewards_window

    for ep in range(episodes):
        try:
            state = env.reset(train_mode=True)
        except Exception as e:
            print(f"[Evaluation] Error resetting environment in eval episode {ep+1}: {e}")
            env.close()
            raise

        episode_reward = 0.0
        done = False

        while not done:
            try:
                # Normalize state if a normalizer is provided.
                norm_state = normalizer.normalize(state) if normalizer is not None else state
                
                # Agent acts deterministically (no exploration noise).
                action = agent.act(norm_state, noise=0.0)
                next_state, reward, done_flags = env.step(action)
            except Exception as e:
                print(f"[Evaluation] Error during step in eval episode {ep+1}: {e}")
                env.close()
                raise

            # For multi-agent: average the rewards across agents.
            episode_reward += np.mean(reward)
            state = next_state
            
            # Consider the episode done when all agents have finished.
            done = all(done_flags)

        evaluation_rewards_window.append(episode_reward)
        print(f"[Evaluation] Eval Episode {ep+1} reward: {episode_reward:.2f}")

    # Compute the sliding window average.
    avg_reward = np.mean(evaluation_rewards_window)
    print(f"[Evaluation] Sliding window average reward (last {len(evaluation_rewards_window)} episodes): {avg_reward:.2f}")

    # Training is considered solved if we have 100 episodes in the window and the average meets the threshold.
    solved = (len(evaluation_rewards_window) == 100 and avg_reward >= threshold)
    return avg_reward, solved

if __name__ == "__main__":
    print("Evaluation module loaded.")
