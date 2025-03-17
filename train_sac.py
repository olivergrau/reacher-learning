"""
training.py

A complete synchronous training loop for SAC on the Unity Reacher environment.
This script ties together the environment wrapper, SACAgent, replay buffer, and
a running state normalizer into one main loop. It includes robust error handling to
ensure the Unity environment is properly shut down if any error occurs.
"""

import os
import time
import numpy as np
import torch
import traceback
from collections import namedtuple

import optuna
from optuna.exceptions import TrialPruned

from codebase.sac.agent import SACAgent
from codebase.utils.normalizer import RunningNormalizer
from codebase.sac.env import BootstrappedEnvironment
from codebase.sac.eval import evaluate
from codebase.replay.replay_buffer import PrioritizedReplay
from codebase.utils.early_stopping import EarlyStopping

from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

# For older numpy versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64

DEBUG = False

def convert_batch_to_tensor(batch, device):
    """
    Converts each field in the sampled batch from numpy arrays to torch tensors.
    
    Args:
        batch: A Transition namedtuple containing numpy arrays.
        device: The torch device to move the tensors to.
    
    Returns:
        A Transition namedtuple where each field is a torch tensor.
    """
    state = torch.as_tensor(batch.state, dtype=torch.float32, device=device)
    action = torch.as_tensor(batch.action, dtype=torch.float32, device=device)
    reward = torch.as_tensor(batch.reward, dtype=torch.float32, device=device)
    next_state = torch.as_tensor(batch.next_state, dtype=torch.float32, device=device)
    mask = torch.as_tensor(batch.mask, dtype=torch.float32, device=device)
    sampling_prob = torch.as_tensor(batch.sampling_prob, dtype=torch.float32, device=device)
    idx = torch.as_tensor(batch.idx, dtype=torch.float32, device=device)
    
    Transition = type(batch) # Remember batch is a namedtuple
    return Transition(state, action, reward, next_state, mask, sampling_prob, idx)

def calculate_num_updates(total_env_steps, env_steps_per_update, updates_per_block):
    """
    Calculate the number of resulting updates given the environment steps per update
    and updates per block.

    Args:
        total_env_steps (int): Total number of environment steps.
        env_steps_per_update (int): Number of environment steps required to trigger an update.
        updates_per_block (int): Number of updates performed per update trigger.

    Returns:
        int: Total number of updates performed.
    """
    if env_steps_per_update <= 0:
        raise ValueError("env_steps_per_update must be greater than 0")
    
    return (total_env_steps // env_steps_per_update) * updates_per_block

def train(
    state_size=33,
    action_size=4,
    episodes=1000,             # Total training episodes
    max_steps=1000,            # Maximum steps per episode
    batch_size=256,            # Batch size for learning
    gamma=0.99,                # Discount factor
    actor_input_size=256,      # Actor input layer size
    actor_hidden_size=256,     # Actor hidden layer size
    critic_input_size=256,     # Critic input layer size
    critic_hidden_size=256,    # Critic hidden layer size
    lr_actor=2e-4,             # Actor learning rate
    lr_critic=2e-4,            # Critic learning rate
    lr_alpha=1e-4,             # Alpha learning rate
    fixed_alpha=None,          # If provided, use this constant alpha instead of learning it.
    target_entropy=-4.0,       # Target entropy for the policy
    tau=0.005,                 # Soft update parameter for target networks         
    replay_capacity=1000000,   # Replay buffer capacity
    eval_frequency=10,         # Evaluate every 10 episodes
    eval_episodes=5,           # Run 5 episodes per evaluation
    eval_threshold=30.0,       # Target average reward to consider environment solved
    unity_worker_id=1,
    use_state_norm=False,      # Enable state normalization
    use_reward_scaling=False,  # Enable reward scaling
    reward_scaling_factor=1.0, # Scaling factor (default: 1.0, i.e. no scaling)
    use_reward_normalization=False,  # Enable reward normalization
    env_steps_per_update=20,   # Number of environment steps to collect before updates
    updates_per_block=10,      # Number of learning updates to perform after env_steps_per_update
    sampling_warm_up_steps=10000,  # Number of steps to sample random actions before using policy
    trial=None                 # Optional Optuna trial for pruning; default is None
):
    LOG_FREQ = 100  # Log metrics every LOG_FREQ steps

    print(f"Training with hyperparameters:")
    print(f"  Episodes: {episodes}")
    print(f"  Max Steps: {max_steps}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Gamma: {gamma}")
    print(f"  Actor Input Size: {actor_input_size}")
    print(f"  Actor Hidden Size: {actor_hidden_size}")
    print(f"  Critic Input Size: {critic_input_size}")
    print(f"  Critic Hidden Size: {critic_hidden_size}")
    print(f"  Actor LR: {lr_actor}")
    print(f"  Critic LR: {lr_critic}")
    print(f"  Alpha LR: {lr_alpha}")
    print(f"  Tau: {tau}")
    print(f"  Replay Capacity: {replay_capacity}")
    print(f"  Evaluation Frequency: {eval_frequency}")
    print(f"  Evaluation Episodes: {eval_episodes}")
    print(f"  Evaluation Threshold: {eval_threshold}")
    print(f"  Use Reward Scaling: {use_reward_scaling}")
    print(f"  Reward Scaling Factor: {reward_scaling_factor}")
    print(f"  Use Reward Normalization: {use_reward_normalization}")
    print(f"  Env Steps per Update Block: {env_steps_per_update}")
    print(f"  Updates per Block: {updates_per_block}")
    print(f"  Use State Normalization: {use_state_norm}")
    print(f"  Sampling Warm-up Steps: {sampling_warm_up_steps}")   

    num_updates = calculate_num_updates(max_steps * episodes, env_steps_per_update, updates_per_block)
    print(f"Total updates if training runs for {episodes} episodes: {num_updates}")

    early_stopping = EarlyStopping(patience=20, min_delta=0.2, verbose=True, zero_patience=10)
    log_dir = os.path.join("runs", "train_sac", time.strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    exe_path = "Reacher_Linux/Reacher.x86_64"

    reward_stats = {"mean": 0.0, "var": 1.0, "count": 0}

    # Number of agents in the environment
    num_agents = 20

    try:
        with BootstrappedEnvironment(exe_path, worker_id=unity_worker_id, use_graphics=False) as env:
            agent = SACAgent(
                state_size=state_size,
                action_size=action_size,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                lr_alpha=lr_alpha,
                fixed_alpha=fixed_alpha,            
                actor_input_size=actor_input_size,
                actor_hidden_size=actor_hidden_size,
                critic_input_size=critic_input_size,
                critic_hidden_size=critic_hidden_size,
                target_entropy=target_entropy,
            )

            replay_kwargs = {
                'memory_size': replay_capacity,
                'batch_size': batch_size,
                'discount': gamma,
                'n_step': 1,
                'history_length': 1
            }

            buffer = PrioritizedReplay(**replay_kwargs)
            normalizer = RunningNormalizer(shape=(state_size,), momentum=0.001) if use_state_norm else None
            
            total_steps = 0
            train_iter = 0  # Counter for learning iterations
            env_steps_since_update = 0  # Counter for environment steps since last update block

            for episode in range(1, episodes + 1):            
                try:                    
                    state = env.reset(train_mode=True)
                except Exception as e:
                    print(f"[Training] Failed to reset environment on episode {episode}: {e}")
                    raise

                episode_reward = 0.0
                env_steps_since_update = 0  # Reset per episode
                
                # Track reward statistics for each agent
                agent_rewards_log = [[] for _ in range(num_agents)]  # List of lists, one per agent

                for step in range(max_steps):
                    total_steps += 1
                    env_steps_since_update += 1                    

                    try:
                        norm_state = normalizer.normalize(state) if normalizer is not None else state
                        
                        # Warm-up: use random actions for the first 'sampling_warm_up_steps'
                        if total_steps < sampling_warm_up_steps:                            
                            action = np.sign(
                                np.random.uniform(
                                    -1, 1, size=(num_agents, action_size))) * np.random.uniform(0.5, 1, size=(num_agents, action_size)
                            )
                            
                            # action = np.clip(np.random.normal(0, 0.8, size=(num_agents, action_size)), -1, 1)                            
                        else:
                            action = agent.act(norm_state, evaluate=False)
                        
                        next_state, reward, done_flags = env.step(action)

                        # Count how many agents got a nonzero reward
                        num_agents_rewarded = np.count_nonzero(np.abs(reward) > 1e-6)  # Use a small threshold

                        # Log if more than one agent received a reward
                        if num_agents_rewarded > 2:
                            print(f"Episode {episode} / Step {step}: {num_agents_rewarded}/{num_agents} agents received a reward!")
                            print(f"Reward Vector: {reward}")

                        # Log rewards per agent
                        for i in range(num_agents):
                            agent_rewards_log[i].append(reward[i])  # Append each agent's reward

                    except Exception as e:
                        print(f"[Training] Error during step {step} in episode {episode}: {e}")
                        raise

                    if normalizer is not None:
                        normalizer.update(state)

                    if use_reward_scaling:
                        reward = np.clip(np.array(reward, dtype=np.float64) * reward_scaling_factor, -1, 1)

                    # --- Reward Normalization --- (Welford's algorithm style)
                    if use_reward_normalization:
                        reward = np.array(reward)
                        n = len(reward)
                        old_count = reward_stats["count"]
                        new_count = old_count + n
                        new_mean = (old_count * reward_stats["mean"] + np.sum(reward)) / new_count
                        new_var = ((old_count * (reward_stats["var"] + reward_stats["mean"]**2) + np.sum(reward**2)) / new_count) - new_mean**2
                        reward_stats["mean"] = new_mean
                        reward_stats["var"] = new_var if new_var > 0 else 1.0
                        reward_stats["count"] = new_count
                        reward = (reward - reward_stats["mean"]) / (np.sqrt(reward_stats["var"]) + 1e-8)
                    
                    episode_reward += np.mean(reward)
                    mask = [0 if d else 1 for d in done_flags]

                    # --------- Learning Phase: Update agent first if buffer is full enough ---------
                    if buffer.size() >= batch_size and env_steps_since_update >= env_steps_per_update:
                        for _ in range(updates_per_block):
                            batch = buffer.sample()

                            # If using prioritized replay, the batch will have an 'idx' field.
                            # Save these indices before converting the rest of the batch to tensors.
                            if hasattr(batch, 'idx'):
                                batch_indices = batch.idx  # Should be an array of indices.
                            else:
                                batch_indices = None
                            
                            batch = convert_batch_to_tensor(batch, device=agent.device)
                            
                            metrics = agent.learn(batch)
                            train_iter += 1
                            
                            # If the replay buffer is prioritized, update its priorities:
                            if batch_indices is not None:
                                # Extract per-sample TD errors (ensure they are positive; add epsilon if needed)
                                td_errors = metrics["td_error"]
                                epsilon = 1e-6  # Prevent zero priority.
                                
                                # Create a list of (index, new_priority) pairs.
                                update_info = [(idx, float(td) + epsilon) for idx, td in zip(batch_indices, td_errors)]
                                buffer.update_priorities(update_info)
                                
                            if train_iter % LOG_FREQ == 0:
                                writer.add_scalar("Loss/Critic", metrics["critic_loss"], train_iter)
                                writer.add_scalar("Loss/Actor", metrics["actor_loss"], train_iter)
                                writer.add_scalar("Loss/Alpha", metrics["alpha_loss"], train_iter)
                                writer.add_scalar("Alpha", metrics["alpha"], train_iter)
                                writer.add_scalar("Q-values/Mean_Q1", metrics["current_Q1_mean"], train_iter)
                                writer.add_scalar("Q-values/Mean_Q2", metrics["current_Q2_mean"], train_iter)
                                writer.add_scalar("Target_Q_Mean", metrics["target_Q_mean"], train_iter)
                                writer.add_scalar("Actor/log_prob_mean", metrics["log_prob_mean"], train_iter)
                                writer.add_scalar("Actor/action_mean", metrics["action_mean"], train_iter)
                                writer.add_scalar("Actor/action_std", metrics["std_mean"], train_iter)
                                writer.add_scalar("Actor/next_log_prob_mean", metrics["next_log_prob_mean"], train_iter)
                                writer.add_scalar("Actor/next_action_mean", metrics["next_action_mean"], train_iter)
                                writer.add_scalar("Actor/next_action_std", metrics["next_std_mean"], train_iter)
                                writer.add_scalar("Training/target_q1_mean", metrics["target_q1_mean"], train_iter)
                                writer.add_scalar("Training/target_q2_mean", metrics["target_q2_mean"], train_iter)
                                writer.add_scalar("Training/min_target_q_mean", metrics["min_target_q_mean"], train_iter)
                                writer.add_scalar("Training/batch_reward_mean", metrics["batch_reward_mean"], train_iter)
                                writer.add_scalar("Training/batch_reward_std", metrics["batch_reward_std"], train_iter)
                                writer.add_scalar("Training/batch_reward_max", metrics["batch_reward_max"], train_iter)
                                writer.add_scalar("Training/batch_reward_min", metrics["batch_reward_min"], train_iter)
                                writer.add_scalar("Training/td_error_mean", metrics["td_error_mean"], train_iter)                            
                        
                        env_steps_since_update = 0  # Reset update counter after learning block

                    # --------- Now, update the replay buffer with the new transitions ---------
                    # Only feed this transition if:
                    #   - The reward vector is not all zeros, OR
                    #   - It is all zeros but with 10% probability (to keep some zero-reward transitions)
                    reward_array = np.array(reward, dtype=np.float64)
                    if np.all(np.abs(reward_array) < 1e-6):
                        if np.random.rand() < 0.1:
                            # With 10% probability, store the zero-reward transition
                            buffer.feed({
                                'state': state,
                                'action': action,
                                'reward': reward,
                                'next_state': next_state,
                                'mask': mask
                            })
                        else:
                            # Otherwise, skip feeding this transition
                            pass
                    else:
                        # If at least one agent got a nonzero reward, store the transition
                        buffer.feed({
                            'state': state,
                            'action': action,
                            'reward': reward,
                            'next_state': next_state,
                            'mask': mask
                        })

                    state = next_state
                    if step % LOG_FREQ == 0:
                        writer.add_scalar("ReplayBuffer/Size", buffer.size(), train_iter)
                        
                    #if all(done_flags):
                    if np.any(done_flags):
                        break                        

                writer.add_scalar("Episode/Reward", episode_reward, episode)
                print(f"Episode {episode:4d} | Average Reward: {episode_reward:7.2f} | Total Steps: {total_steps} | Total updates: {train_iter}")

                if episode % eval_frequency == 0 and episode > 40:
                    print(f"Evaluating agent at episode {episode}...")
                    try:
                        avg_reward, solved = evaluate(agent, env, normalizer=normalizer,
                                                      episodes=eval_episodes, threshold=eval_threshold)
                        
                        writer.add_scalar("Eval/AverageReward", avg_reward, episode)
                        print(f"Evaluation at training episode {episode}: Sliding window avg reward = {avg_reward:.2f}")
                        
                        if trial is not None:
                            report_step_reward = episode
                            trial.report(-avg_reward, report_step_reward)
                            if trial.should_prune():
                                print(f"Trial pruned at episode {episode} due to insufficient evaluation reward improvement.")
                                raise TrialPruned()

                        if solved:
                            print("Environment solved! Stopping training early.")
                            new_weights = extract_agent_weights(agent)
                            save_path = "sac_agent_weights_solved.pth"
                            torch.save(new_weights, save_path)
                            print(f"Saved agent weights to {save_path}")
                            break                        
                            
                        if early_stopping.step(avg_reward):
                            print("Early stopping triggered (evaluation reward). Stopping training.")
                            break
                        
                    except Exception as eval_e:
                        print(f"[Training] Evaluation failed at episode {episode}: {eval_e}")
    
    except Exception as e:
        print(f"[Training] An error occurred during training: {e}")
        traceback.print_exc()
    
    finally:
        try:
            env.close()
        except Exception:
            pass
        
        writer.close()
        print("[Training] Training process terminated. Unity environment shut down.")
    
    return avg_reward


def extract_agent_weights(agent):
    return {
        "actor": {k: v.cpu().detach() for k, v in agent.actor.state_dict().items()},
        "critic": {k: v.cpu().detach() for k, v in agent.critic.state_dict().items()},
        "critic_target": {k: v.cpu().detach() for k, v in agent.critic_target.state_dict().items()},
        "log_alpha": agent.log_alpha.cpu().detach()
    }


if __name__ == "__main__":
    print("Training module SAC loaded. Starting manual training run.")
    
    avg_reward = train(
        state_size=33,
        action_size=4,
        episodes=1000,
        max_steps=1000,
        batch_size=256,
        gamma=0.95,
        actor_input_size=256,
        actor_hidden_size=256,
        critic_input_size=256,
        critic_hidden_size=256,
        lr_actor=1e-4,
        lr_critic=3e-5,
        lr_alpha=1e-5,
        target_entropy=-4,
        tau=0.003,
        replay_capacity=int(1e6),
        eval_frequency=10,
        eval_episodes=1,
        eval_threshold=30.0,
        unity_worker_id=1,
        use_state_norm=False,
        use_reward_scaling=True,
        reward_scaling_factor=15.0,
        use_reward_normalization=True,
        env_steps_per_update=20,
        updates_per_block=8,
        sampling_warm_up_steps=10000
    )
    print(f"Training completed. Average reward: {avg_reward:.2f}")
