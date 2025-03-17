import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import optuna
from train_ddpg import train
import time

def run_trial(q, params):
    # Execute the training function and put the result in the queue.
    result = train(**params)
    q.put(result)

def coarse_objective(trial):
    # Coarse search over the most sensitive parameters.
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 1e-3, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-5, 1e-3, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
    exploration_noise = trial.suggest_float("exploration_noise", 0.1, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    critic_clip = trial.suggest_float("critic_clip", 5.0, 10.0)
    critic_weight_decay = trial.suggest_float("critic_weight_decay", 1e-6, 1e-3, log=True)
    
    use_ou_noise = trial.suggest_categorical("use_ou_noise", [False, True])
    use_state_norm = trial.suggest_categorical("use_state_norm", [False, True])
    
    env_steps_per_update = trial.suggest_int("env_steps_per_update", 10, 50, step=10)
    update_multiplier = trial.suggest_float("update_multiplier", 0.1, 1.0)
    updates_per_block = int(env_steps_per_update * update_multiplier)

    # Fixed (coarse) hyperparameters.
    gamma = 0.98
    replay_capacity = int(1e6)
    eval_frequency = 10       # Evaluate less frequently for faster search.
    eval_episodes = 5
    eval_threshold = 30.0

    # Use fixed reward transforms.
    use_reward_scaling = False
    reward_scaling_factor = 1.0
    use_reward_normalization = False

    params = {
        "trial": trial,
        "state_size": 33,
        "action_size": 4,
        "episodes": 50,          # Fewer episodes for fast coarse search.
        "max_steps": 1000,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "critic_clip": critic_clip,
        "tau": tau,
        "use_state_norm": use_state_norm,
        "use_ou_noise": use_ou_noise,
        "ou_noise_theta": 0.15,
        "ou_noise_sigma": 0.08,
        "initial_noise_scaling_factor": exploration_noise,
        "min_noise_scaling_factor": 0.05,
        "noise_decay_rate": 0.99,
        "replay_capacity": replay_capacity,
        "critic_weight_decay": critic_weight_decay,
        "eval_frequency": eval_frequency,
        "eval_episodes": eval_episodes,
        "eval_threshold": eval_threshold,
        "unity_worker_id": 0,  # We can reuse worker_id=0 in coarse phase.
        "use_reward_scaling": use_reward_scaling,
        "reward_scaling_factor": reward_scaling_factor,
        "use_reward_normalization": use_reward_normalization,
        "env_steps_per_update": env_steps_per_update,
        "updates_per_block": updates_per_block
    }
    q = mp.Queue()
    p = mp.Process(target=run_trial, args=(q, params))
    p.start()
    p.join()
    avg_reward = q.get()

    # Return negative reward for minimization.
    return -avg_reward

def fine_objective(trial):
    # The best parameters from the coarse phase (will be set externally).
    # For each parameter, we narrow the search range around the coarse best value.
    lr_actor = trial.suggest_float("lr_actor", best_coarse["lr_actor"] * 0.5, best_coarse["lr_actor"] * 1.5, log=True)
    lr_critic = trial.suggest_float("lr_critic", best_coarse["lr_critic"] * 0.5, best_coarse["lr_critic"] * 1.5, log=True)
    tau = trial.suggest_float("tau", best_coarse["tau"] * 0.8, best_coarse["tau"] * 1.2, log=True)
    exploration_noise = trial.suggest_float("exploration_noise", best_coarse["initial_noise_scaling_factor"] * 0.8, best_coarse["initial_noise_scaling_factor"] * 1.2)
    batch_size = trial.suggest_categorical("batch_size", [best_coarse["batch_size"]])
    critic_clip = trial.suggest_float("critic_clip", best_coarse["critic_clip"] * 0.8, best_coarse["critic_clip"] * 1.2)
    critic_weight_decay = trial.suggest_float("critic_weight_decay", best_coarse["critic_weight_decay"] * 0.5, best_coarse["critic_weight_decay"] * 1.5, log=True)

    # For fine tuning we can keep these fixed.
    gamma = 0.98
    replay_capacity = int(1e6)
    eval_frequency = 10
    eval_episodes = 5
    eval_threshold = 30.0
    use_reward_scaling = best_coarse["use_reward_scaling"]
    reward_scaling_factor = best_coarse["reward_scaling_factor"]
    use_reward_normalization = best_coarse["use_reward_normalization"]
    env_steps_per_update = best_coarse["env_steps_per_update"]
    updates_per_block = best_coarse["updates_per_block"]

    use_state_norm = best_coarse["use_state_norm"]
    use_ou_noise = best_coarse["use_ou_noise"]

    params = {
        "trial": trial,
        "state_size": 33,
        "action_size": 4,
        "episodes": 50,         # Use the same episode count for consistency.
        "max_steps": 1000,
        "batch_size": batch_size,
        "gamma": gamma,
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "critic_clip": critic_clip,
        "tau": tau,
        "use_state_norm": use_state_norm,
        "use_ou_noise": use_ou_noise,
        "ou_noise_theta": 0.15,
        "ou_noise_sigma": 0.08,
        "initial_noise_scaling_factor": exploration_noise,
        "min_noise_scaling_factor": 0.05,
        "noise_decay_rate": 0.99,
        "replay_capacity": replay_capacity,
        "critic_weight_decay": critic_weight_decay,
        "eval_frequency": eval_frequency,
        "eval_episodes": eval_episodes,
        "eval_threshold": eval_threshold,
        "unity_worker_id": 0,  # For fine phase, we can continue with worker_id=0.
        "use_reward_scaling": use_reward_scaling,
        "reward_scaling_factor": reward_scaling_factor,
        "use_reward_normalization": use_reward_normalization,
        "env_steps_per_update": env_steps_per_update,
        "updates_per_block": updates_per_block
    }
    q = mp.Queue()
    p = mp.Process(target=run_trial, args=(q, params))
    p.start()
    p.join()
    avg_reward = q.get()
    return -avg_reward

if __name__ == "__main__":
    # Phase 1: Coarse Search
    coarse_study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    coarse_study.optimize(coarse_objective, n_trials=50)
    
    best_coarse = coarse_study.best_trial.params
    print("Coarse phase best parameters:")
    for key, value in best_coarse.items():
        print(f"  {key}: {value}")
        
    # Phase 2: Fine Search. The best coarse parameters will be used to narrow the search space.
    fine_study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    fine_study.optimize(fine_objective, n_trials=50)
    
    best_fine = fine_study.best_trial.params
    print("Fine phase best parameters:")
    for key, value in best_fine.items():
        print(f"  {key}: {value}")
