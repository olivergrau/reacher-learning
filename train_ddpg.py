"""
training.py

A complete synchronous training loop for DDPG on the Unity Reacher environment.
This script ties together the environment wrapper, DDPGAgent, replay buffer, and
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

from codebase.ddpg.agent import DDPGAgent
from codebase.utils.normalizer import RunningNormalizer
from codebase.ddpg.env import EnvWrapper
from codebase.ddpg.eval import evaluate
from codebase.replay.replay_buffer import UniformReplay
from codebase.utils.early_stopping import EarlyStopping

from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

# For older numpy versions
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64


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
    
    Transition = type(batch) # Remember batch is a namedtuple
    return Transition(state, action, reward, next_state, mask)


def train(
    state_size=33,
    action_size=4,
    episodes=1000,             # Total training episodes
    max_steps=1000,            # Maximum steps per episode
    batch_size=256,            # Batch size for learning
    gamma=0.99,                # Discount factor
    actor_input_size=400,      # Actor input layer size
    actor_hidden_size=300,     # Actor hidden layer size
    critic_input_size=400,     # Critic input layer size
    critic_hidden_size=300,    # Critic hidden layer size
    lr_actor=2e-4,             # Actor learning rate
    lr_critic=2e-4,            # Critic learning rate
    critic_clip=None,          # Gradient clipping for critic
    critic_weight_decay=1e-4,  # L2 weight decay for critic
    tau=0.005,                 # Soft update parameter for target networks
    use_ou_noise=True,         # Enable Ornstein-Uhlenbeck noise      
    ou_noise_theta=0.15,       # Noise theta parameter
    ou_noise_sigma=0.2,        # Noise sigma parameter  
    initial_noise_scaling_factor=0.3,  # Initial exploration noise factor
    min_noise_scaling_factor=0.05,     # Minimum noise scaling factor after decay
    noise_decay_rate=0.99,     # Decay rate per episode for noise scaling factor
    replay_capacity=1000000,    # Replay buffer capacity
    eval_frequency=10,         # Evaluate every 10 episodes
    eval_episodes=5,           # Run 5 episodes per evaluation
    eval_threshold=30.0,       # Target average reward to consider environment solved
    unity_worker_id=1,
    use_state_norm=False,       # Enable state normalization
    use_reward_scaling=False,  # Enable reward scaling
    reward_scaling_factor=1.0, # Scaling factor (default: 1.0, i.e. no scaling)
    use_reward_normalization=False,  # Enable reward normalization
    env_steps_per_update=20,   # Number of environment steps to collect before updates
    updates_per_block=10,      # Number of learning updates to perform after env_steps_per_update
    trial=None               # Optional Optuna trial for pruning; default is None
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
    print(f"  Critic Clip: {critic_clip}")
    print(f"  Critic Weight Decay: {critic_weight_decay}")
    print(f"  Tau: {tau}")
    print(f"  Initial Noise Scaling Factor: {initial_noise_scaling_factor}")
    print(f"  Min Noise Scaling Factor: {min_noise_scaling_factor}")
    print(f"  Noise Decay Rate: {noise_decay_rate}")
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
    print(f"  Use Ornstein-Uhlenbeck Noise: {use_ou_noise}")
    print(f"  OU Noise Theta: {ou_noise_theta}")
    print(f"  OU Noise Sigma: {ou_noise_sigma}")    

    # Early stopping based on evaluation reward.
    early_stopping = EarlyStopping(patience=20, min_delta=0.2, verbose=True)

    # TensorBoard logging directory.
    log_dir = os.path.join("runs", "train_ddpg", time.strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    exe_path = "Reacher_Linux/Reacher.x86_64"
    
    # Variables for actor loss plateau detection.
    previous_avg_actor_loss = None
    plateau_counter = 0
    plateau_threshold = 1e-3  # If the change is less than this, actor loss is considered plateaued.
    actor_loss_window = []       # List to store recent actor losses.
    actor_loss_window_size = 10  # Number of recent updates to average.
    actor_loss_counter = 0       # Counter for insufficient actor loss (if too high).
    actor_loss_patience = 10      # Number of evaluations to tolerate insufficient actor loss.
    plateau_patience = 20
    min_actor_threshold = -0.2   # Minimum (more negative) threshold for acceptable actor loss.

    # For reward normalization, we use a simple running statistic.
    reward_stats = {"mean": 0.0, "var": 1.0, "count": 0}

    try:
        with EnvWrapper(exe_path, worker_id=unity_worker_id, use_graphics=False) as env:
            agent = DDPGAgent(
                state_size=state_size,
                action_size=action_size,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                critic_clip=critic_clip,
                critic_weight_decay=critic_weight_decay,
                use_ou_noise=use_ou_noise,
                ou_noise_theta=ou_noise_theta,
                ou_noise_sigma=ou_noise_sigma,
                actor_input_size=actor_input_size,
                actor_hidden_size=actor_hidden_size,
                critic_input_size=critic_input_size,
                critic_hidden_size=critic_hidden_size,
            )

            replay_kwargs = {
                'memory_size': replay_capacity,
                'batch_size': batch_size,
                'discount': gamma,
                'n_step': 1,
                'history_length': 1
            }

            buffer = UniformReplay(**replay_kwargs)
            normalizer = RunningNormalizer(shape=(state_size,), momentum=0.001) if use_state_norm else None
            
            total_steps = 0
            train_iter = 0  # Counter for learning iterations
            env_steps_since_update = 0  # Counter for environment steps since last update block
            
            # Set initial noise scaling factor
            current_noise_scaling = initial_noise_scaling_factor

            for episode in range(1, episodes + 1):
                # Decay noise scaling factor per episode.
                current_noise_scaling = max(min_noise_scaling_factor,
                                            initial_noise_scaling_factor * (noise_decay_rate ** episode))

                writer.add_scalar("Env/Exploration_Noise", current_noise_scaling, episode)

                try:
                    state = env.reset(train_mode=True)
                except Exception as e:
                    print(f"[Training] Failed to reset environment on episode {episode}: {e}")
                    raise

                episode_reward = 0.0
                env_steps_since_update = 0  # Reset per episode

                agent.reset_noise()
                for step in range(max_steps):
                    total_steps += 1
                    env_steps_since_update += 1                    

                    try:
                        norm_state = normalizer.normalize(state) if normalizer is not None else state
                        action = agent.act(norm_state, noise=current_noise_scaling)
                        next_state, reward, done_flags = env.step(action)
                    except Exception as e:
                        print(f"[Training] Error during step {step} in episode {episode}: {e}")
                        raise

                    if normalizer is not None:
                        normalizer.update(state)

                    # --- Reward Scaling ---
                    if use_reward_scaling:
                        reward = np.array(reward) * reward_scaling_factor

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
                    
                    if all(done_flags):
                        break

                    # Instead of updating at every timestep, update after env_steps_per_update steps
                    if env_steps_since_update >= env_steps_per_update and buffer.size() >= batch_size:                        
                        for _ in range(updates_per_block):
                            batch = buffer.sample()
                            batch = convert_batch_to_tensor(batch, device=agent.device)
                            metrics = agent.learn(batch)
                            train_iter += 1
                            
                            if train_iter % LOG_FREQ == 0:
                                writer.add_scalar("Loss/Critic", metrics["critic_loss"], train_iter)
                                writer.add_scalar("Loss/Actor", metrics["actor_loss"], train_iter)
                                writer.add_scalar("Q-values/Current", metrics["current_Q_mean"], train_iter)
                                writer.add_scalar("Q-values/Target", metrics["target_Q_mean"], train_iter)
                            
                            actor_loss_window.append(metrics["actor_loss"])
                            
                            if len(actor_loss_window) > actor_loss_window_size:
                                actor_loss_window = actor_loss_window[-actor_loss_window_size:]
                        
                        env_steps_since_update = 0  # Reset the counter after the update block
    
                writer.add_scalar("Episode/Reward", episode_reward, episode)
                print(f"Episode {episode:4d} | Average Reward: {episode_reward:7.2f} | Total Steps: {total_steps}")

                # Evaluation block every eval_frequency episodes
                if episode % eval_frequency == 0:
                    try:
                        avg_reward, solved = evaluate(agent, env, normalizer=normalizer,
                                                      episodes=eval_episodes, threshold=eval_threshold)  # in evaluation, no reward scaling/norm is necessary
                        
                        writer.add_scalar("Eval/AverageReward", avg_reward, episode)
                        print(f"Evaluation at training episode {episode}: Sliding window avg reward = {avg_reward:.2f}")
                        
                        # Compute average actor loss from the sliding window.
                        if len(actor_loss_window) > 0:
                            avg_actor_loss = np.mean(actor_loss_window)
                        else:
                            avg_actor_loss = float('inf')
                        
                        print(f"Average actor loss over last {actor_loss_window_size} updates: {avg_actor_loss:.4f}")

                        # --- Pruning: Report one metric using different step numbers ---
                        if trial is not None:
                            # Report negative evaluation reward (since higher reward is better) at an even step.
                            report_step_reward = episode # * 2
                            trial.report(-avg_reward, report_step_reward)
                            if trial.should_prune():
                                print(f"Trial pruned at episode {episode} due to insufficient evaluation reward improvement.")
                                raise TrialPruned()

                        # First, check if actor loss is insufficient (too high).
                        if avg_actor_loss > 0 or avg_actor_loss > min_actor_threshold:
                            actor_loss_counter += 1
                            print(f"Actor loss insufficient. Counter: {actor_loss_counter} / {actor_loss_patience}")
                        else:
                            actor_loss_counter = 0

                        # Now, check for plateau in actor loss.
                        if previous_avg_actor_loss is not None:
                            if abs(avg_actor_loss - previous_avg_actor_loss) < plateau_threshold:
                                plateau_counter += 1
                                print(f"Actor loss plateau detected. Plateau counter: {plateau_counter}")
                            else:
                                plateau_counter = 0
                        previous_avg_actor_loss = avg_actor_loss
                        
                        # If plateau persists over plateau_patience evaluations, early stop.
                        if plateau_counter >= plateau_patience:
                            print(f"Actor loss has plateaued over {plateau_patience} evaluations. Early stopping triggered.")
                            break

                        if solved:
                            print("Environment solved! Stopping training early.")
                            new_weights = extract_agent_weights(agent)
                            save_path = "ddpg_agent_weights_solved.pth"
                            torch.save(new_weights, save_path)
                            print(f"Saved agent weights to {save_path}")
                            break

                        if actor_loss_counter >= actor_loss_patience:
                            print("Actor loss early stopping triggered. Stopping training.")
                            break
                            
                        if early_stopping.step(avg_reward):
                            print("Early stopping triggered (evaluation reward). Stopping training.")
                            break
                        
                    except Exception as eval_e:
                        print(f"[Training] Evaluation failed at episode {episode}: {eval_e}")
                        # Optionally, decide whether to continue or break.
    
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
        "actor": {k: v.cpu() for k, v in agent.actor.state_dict().items()},
        "actor_target": {k: v.cpu() for k, v in agent.actor_target.state_dict().items()},
        "critic": {k: v.cpu() for k, v in agent.critic.state_dict().items()},
        "critic_target": {k: v.cpu() for k, v in agent.critic_target.state_dict().items()},
    }


if __name__ == "__main__":
    print("Training module DDPG loaded. Starting manual training run.")

    avg_reward = train(
        state_size=33,
        action_size=4,
        episodes=500,             # Total training episodes
        max_steps=1000,            # Maximum steps per episode
        batch_size=256,            # Batch size for learning
        gamma=0.95,                # Discount factor
        actor_input_size=128,      # Actor input layer size
        actor_hidden_size=256,     # Actor hidden layer size
        critic_input_size=128,     # Critic input layer size
        critic_hidden_size=256,    # Critic hidden layer size
        lr_actor=1e-4,             # Actor learning rate
        lr_critic=3e-4,            # Critic learning rate
        critic_clip=10,            # Gradient clipping for critic
        critic_weight_decay=0.0,   # L2 weight decay for critic
        tau=0.003,                 # Soft update parameter for target networks
        use_ou_noise=False,         # Enable Ornstein-Uhlenbeck noise      
        ou_noise_theta=0.15,       # Noise theta parameter
        ou_noise_sigma=0.08,        # Noise sigma parameter  
        initial_noise_scaling_factor=0.3,  # Initial exploration noise factor
        min_noise_scaling_factor=0.05,     # Minimum noise scaling factor after decay
        noise_decay_rate=0.99,     # Decay rate per episode for noise scaling factor
        replay_capacity=int(1e6),       # Replay buffer capacity
        eval_frequency=5,         # Evaluate every 10 episodes
        eval_episodes=5,           # Run 5 episodes per evaluation
        eval_threshold=30.0,       # Target average reward to consider environment solved
        unity_worker_id=1,
        use_state_norm=False,      # Enable state normalization
        use_reward_scaling=False,  # Enable reward scaling
        reward_scaling_factor=1.0, # Scaling factor (default: 1.0, i.e. no scaling)
        use_reward_normalization=False,  # Enable reward normalization
        env_steps_per_update=25,   # Number of environment steps to collect before updates (e.g., 20)
        updates_per_block=9       # Number of learning updates to perform after env_steps_per_update (e.g., 10)
    )

    print(f"Training completed. Average reward: {avg_reward:.2f}")
