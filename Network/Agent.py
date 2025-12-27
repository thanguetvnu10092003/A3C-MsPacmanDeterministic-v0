import torch
import torch.nn.functional as F
import numpy as np
from Network.NN import Network


class Agent():
    """
    Improved A3C Agent with better exploration and training stability
    Optimized for 4GB GPU
    """

    def __init__(self, action_size, input_channels=4):
        # Hyperparameters - tuned for better exploration
        self.LEARNING_RATE = 3e-4  # Increased from 1e-4
        self.DISCOUNT_FACTOR = 0.99
        self.ENTROPY_COEF = 0.01  # Increased from 0.001 for better exploration
        self.VALUE_COEF = 0.5
        self.MAX_GRAD_NORM = 0.5  # Gradient clipping
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(action_size, input_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LEARNING_RATE)
        
        # Learning rate scheduler - reduce LR over time
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50000, gamma=0.5
        )
        
        # Training statistics
        self.training_step = 0

    def save_model(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_step': self.training_step
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'network_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['network_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'training_step' in checkpoint:
                self.training_step = checkpoint['training_step']
        else:
            # Backward compatibility with old model format
            self.network.load_state_dict(checkpoint)

    def act(self, state):
        self.network.eval()
        with torch.no_grad():
            if state.ndim == 3:
                state = [state]

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            action_values, _ = self.network(state)
            policy = F.softmax(action_values, dim=-1)
            actions = np.array([np.random.choice(len(p), p=p) for p in policy.cpu().numpy()])
        self.network.train()
        return actions

    def step(self, state, action, reward, next_state, done):
        batch_size = state.shape[0]

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).to(dtype=torch.float32)

        action_values, state_value = self.network(state)
        _, next_state_value = self.network(next_state)

        # TD target and advantage
        target_state_value = reward + self.DISCOUNT_FACTOR * next_state_value * (1 - done)
        advantage = target_state_value - state_value
        
        # Policy probabilities
        probs = F.softmax(action_values, dim=-1)
        logprobs = F.log_softmax(action_values, dim=-1)

        # Entropy for exploration (higher = more random actions)
        entropy = -torch.sum(probs * logprobs, axis=-1)
        
        batch_idx = np.arange(batch_size)
        logp_actions = logprobs[batch_idx, action]

        # Actor loss with entropy bonus
        actor_loss = -(logp_actions * advantage.detach()).mean()
        entropy_loss = -self.ENTROPY_COEF * entropy.mean()
        
        # Critic loss
        critic_loss = self.VALUE_COEF * F.mse_loss(target_state_value.detach(), state_value)
        
        # Total loss
        total_loss = actor_loss + entropy_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.MAX_GRAD_NORM)
        
        self.optimizer.step()
        self.scheduler.step()
        self.training_step += 1
        
        # Return metrics for logging
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item()
        }
