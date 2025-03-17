import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from codebase.sac.net.actor import Actor  # our SAC actor network
from codebase.sac.net.critic import Critic  # our SAC critic network

DEBUG = False  # Set to True for detailed debugging
PRINT_EVERY = 1  # Print debug information every PRINT_EVERY steps
MAX_PRINT_STEPS = 50  # Maximum number of steps to print debug information

class SACAgent:
    def __init__(
        self,
        state_size=33,
        action_size=4,
        lr_actor=1e-3,
        lr_critic=1e-3,
        actor_input_size=400,      # Actor input layer size
        actor_hidden_size=300,     # Actor hidden layer size
        critic_input_size=400,     # Critic input layer size
        critic_hidden_size=300,    # Critic hidden layer size
        lr_alpha=1e-4,              # Learning rate for alpha (temperature parameter)
        fixed_alpha=None,         # If provided, use this constant alpha instead of learning it.
        gamma=0.99,
        tau=0.005,
        target_entropy=None,
        device=None,
        label="SACAgent",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_id = "SACAgent (" + label + ")"
        self.total_it = 0

        print(f"\n{self.agent_id}: Using device: {self.device}")
        print(f"{self.agent_id}: Using gamma: {gamma}")
        print(f"{self.agent_id}: Using tau: {tau}")
        print(f"{self.agent_id}: Using actor input size: {actor_input_size}")
        print(f"{self.agent_id}: Using actor hidden size: {actor_hidden_size}")
        print(f"{self.agent_id}: Using critic input size: {critic_input_size}")
        print(f"{self.agent_id}: Using critic hidden size: {critic_hidden_size}")
        print(f"{self.agent_id}: Using actor learning rate: {lr_actor}")
        print(f"{self.agent_id}: Using critic learning rate: {lr_critic}")
        print(f"{self.agent_id}: Using alpha learning rate: {lr_alpha}")
        print(f"{self.agent_id}: Using static alpha: {fixed_alpha}")
        print(f"{self.agent_id}: Using target entropy: {target_entropy}")
        print()

        # Initialize SAC Actor network and its optimizer
        self.actor = Actor(
            state_size, action_size,
            hidden1=actor_input_size,
            hidden2=actor_hidden_size
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Initialize SAC Critic network and its target network
        self.critic = Critic(
            state_size, action_size,
            hidden1=critic_input_size,
            hidden2=critic_hidden_size
        ).to(self.device)

        self.critic_target = Critic(
            state_size, action_size,
            hidden1=critic_input_size,
            hidden2=critic_hidden_size
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Set up temperature parameter.
        # If static_alpha is provided, we use that value and do not learn alpha.
        self.fixed_alpha = fixed_alpha
        if self.fixed_alpha is None:
            # Learn log_alpha to ensure alpha is always positive.
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None

        # Set target entropy (default: -action_size)
        self.target_entropy = target_entropy if target_entropy is not None else -action_size

        print(f"{self.agent_id}: Initialized SAC Agent with target entropy: {self.target_entropy}")

    def act(self, state, evaluate=False):
        """
        Given a state, select an action.
        When evaluate=True, returns a deterministic action (tanh(mean)).
        Otherwise, samples stochastically from the policy.
        """
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            return action.clamp(-1, 1).detach().cpu().numpy()
        else:
            action, _, _, _ = self.actor.sample(state)
            return action.clamp(-1, 1).detach().cpu().numpy()

    def learn(self, batch):
        """
        Update the SAC agent based on a batch of transitions.
        The batch is expected to be a namedtuple or similar structure containing:
          - state: shape [batch_size, state_size]
          - action: shape [batch_size, action_size]
          - reward: shape [batch_size]
          - next_state: shape [batch_size, state_size]
          - mask: shape [batch_size] (1 if not done, 0 if done)
        Returns a dictionary of metrics for logging.
        """
        
        self.total_it += 1

        # Unpack the batch and move to the appropriate device.
        state = batch.state.to(self.device)
        action = batch.action.to(self.device)
        reward = batch.reward.unsqueeze(1).to(self.device)
        next_state = batch.next_state.to(self.device)
        mask = batch.mask.unsqueeze(1).to(self.device)

        reward_log = reward.mean().item(), reward.std().item(), reward.min().item(), reward.max().item()

        if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
            print(f"\n--- Learning Iteration {self.total_it} ---")
            print(f"Batch state shape: {state.shape}")
            print(f"Batch action shape: {action.shape}")
            print(f"Batch reward shape: {reward.shape}")
            print(f"Batch next_state shape: {next_state.shape}")
            print(f"Batch reward mean: {reward.mean().item():.4f}, std: {reward.std().item():.4f}")
            print(f"Batch mask mean: {mask.float().mean().item():.4f}")
                
        # ----------------------
        # Critic Update
        # ----------------------
        with torch.no_grad():
            # Sample action for next state and compute its log probability.
            next_action, next_log_prob, _, next_std = self.actor.sample(next_state)
            if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
                print(f"Next action mean (sample): {next_action.mean().item():.4f}")
                print(f"Next log_prob mean: {next_log_prob.mean().item():.4f}")
                print(f"Next std mean: {next_std.mean().item():.4f}")

            # Evaluate target Q-values using the target critic network.
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
                print(f"Target Q1 mean: {target_Q1.mean().item():.4f}, Target Q2 mean: {target_Q2.mean().item():.4f}")
                print(f"Min Target Q mean: {target_Q.mean().item():.4f}")

            # Use learned alpha if static_alpha is not provided.
            alpha = self.log_alpha.exp() if self.fixed_alpha is None else self.fixed_alpha
            if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
                print(f"Alpha value: {alpha.item() if isinstance(alpha, torch.Tensor) else alpha}")

            # Compute the entropy-adjusted target value.
            target_value = target_Q - alpha * next_log_prob
            target = reward + mask * self.gamma * target_value

            if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
                print(f"Reward mean: {reward.mean().item():.4f}")
                print(f"Target value mean: {target_value.mean().item():.4f}")
                print(f"Final target mean: {target.mean().item():.4f}")

        # Current Q estimates from the critic.
        current_Q1, current_Q2 = self.critic(state, action)
        critic1_loss = F.mse_loss(current_Q1, target)
        critic2_loss = F.mse_loss(current_Q2, target)
        critic_loss = critic1_loss + critic2_loss

        if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
            print(f"Critic1 loss: {critic1_loss.item():.4f}, Critic2 loss: {critic2_loss.item():.4f}")
            print(f"Total critic loss: {critic_loss.item():.4f}")

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_optimizer.step()

        # ----------------------
        # Actor Update
        # ----------------------
        # Sample actions from the current policy.
        sampled_action, log_prob, _, std = self.actor.sample(state)
        
        # Evaluate the Q-value for these actions.
        Q1, Q2 = self.critic(state, sampled_action)
        Q = torch.min(Q1, Q2)

        # Actor loss: minimize (alpha * log_prob - Q)
        actor_loss = (alpha * log_prob - Q).mean()

        if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
            print(f"Sampled action mean: {sampled_action.mean().item():.4f}")
            print(f"Log_prob mean: {log_prob.mean().item():.4f}")
            print(f"Q value mean: {Q.mean().item():.4f}")
            print(f"Actor loss: {actor_loss.item():.4f}")
            print(f"Std mean: {std.mean().item():.4f}")

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------
        # Temperature (alpha) Update (Only if learned)
        # ----------------------
        if self.fixed_alpha is None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
                print(f"Alpha loss before update: {alpha_loss.item():.4f}")
                print(f"Log_alpha before update: {self.log_alpha.item():.4f}")
                print(f"Log_prob mean for alpha update: {log_prob.mean().item():.4f}")
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()

            if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
                print(f"Alpha gradient: {self.log_alpha.grad.item():.4f}")
            
            if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
                print(f"Log_alpha before clamping: {self.log_alpha.item():.4f}")
            
            self.log_alpha.data.clamp_(-10, 2)

            if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
                print(f"Log_alpha after clamping: {self.log_alpha.item():.4f}")
            
            self.alpha_optimizer.step()
        else:
            alpha_loss = 0.0

        # ----------------------
        # Soft Update of Critic Target Networks
        # ----------------------
        self.soft_update(self.critic, self.critic_target)

        # ----------------------
        # Compute TD Error for Prioritized Replay
        # ----------------------
        with torch.no_grad():
            # Compute the absolute differences between each critic's prediction and the target.
            td_error1 = torch.abs(current_Q1 - target)
            td_error2 = torch.abs(current_Q2 - target)
            
            # Average the two to get a single TD error per sample.
            td_error = (td_error1 + td_error2) / 2.0
            
            # Remove the extra dimension so each sample's error is a scalar.
            td_error = td_error.squeeze(1)
            
        metrics = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            "alpha": alpha.item() if isinstance(alpha, torch.Tensor) else alpha,
            "current_Q1_mean": current_Q1.mean().item(),
            "current_Q2_mean": current_Q2.mean().item(),
            "target_Q_mean": target.mean().item(),
            "action_mean": sampled_action.mean().item(),
            "log_prob_mean": log_prob.mean().item() if log_prob is not None else 0.0,
            "std_mean": std.mean().item(),
            "next_action_mean": next_action.mean().item(),
            "next_log_prob_mean": next_log_prob.mean().item() if next_log_prob is not None else 0.0,
            "next_std_mean": next_std.mean().item(),
            "target_q1_mean": target_Q1.mean().item(),
            "target_q2_mean": target_Q2.mean().item(),
            "min_target_q_mean": target_Q.mean().item(),
            "td_error_mean": td_error.mean().item(),
            "td_error": td_error.cpu().numpy(),  # Array of TD errors per sample,
            "batch_reward_mean": reward_log[0],
            "batch_reward_std": reward_log[1],
            "batch_reward_min": reward_log[2],
            "batch_reward_max": reward_log[3],
        }
        
        if DEBUG and self.total_it % PRINT_EVERY == 0 and self.total_it < MAX_PRINT_STEPS:
            print(f"\n--- Learning Iteration ended {self.total_it} ---")

        return metrics

    def soft_update(self, net, target_net):
        """
        Perform Polyak averaging to update the target network:
        target_param = tau * param + (1 - tau) * target_param
        """
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
