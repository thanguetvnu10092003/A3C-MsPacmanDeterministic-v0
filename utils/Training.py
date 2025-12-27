from utils.Multiple_Env import EnvBatch
from utils.Evaluate_Reward import evaluate
import numpy as np
import tqdm


class TrainingModel:
    def __init__(self, env, training_epochs, number_of_envs=8):
        """
        Args:
            env: Environment for evaluation
            training_epochs: Number of training steps
            number_of_envs: Number of parallel environments (8 for 4GB GPU)
        """
        self.env = env
        self.env_batch = EnvBatch(number_of_envs)
        self.training_epochs = training_epochs

    def train(self, agent):
        batch_states = self.env_batch.reset()
        
        # Tracking metrics
        best_reward = 0
        recent_rewards = []
        
        with tqdm.trange(0, self.training_epochs) as progress_bar:
            for i in progress_bar:
                batch_actions = agent.act(batch_states)
                batch_next_states, batch_rewards, batch_dones, _ = self.env_batch.step(batch_actions)
                batch_rewards *= 0.01  # Reward scaling

                # Training step with metrics
                metrics = agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
                batch_states = batch_next_states

                # Update progress bar with metrics
                if i % 100 == 0:
                    progress_bar.set_postfix({
                        'entropy': f"{metrics['entropy']:.3f}",
                        'loss': f"{metrics['total_loss']:.3f}"
                    })

                # Evaluation every 5000 steps (more frequent than before)
                if i % 5000 == 0 and i > 0:
                    rewards = evaluate(agent, self.env, n_episodes=5)
                    average_reward = np.mean(rewards)
                    recent_rewards.append(average_reward)
                    
                    print(f"\n[Step {i}] Average reward: {average_reward:.1f} | Entropy: {metrics['entropy']:.3f}")
                    
                    # Save if best performance
                    if average_reward > best_reward:
                        best_reward = average_reward
                        model_path = f"models/model_best.pth"
                        agent.save_model(model_path)
                        print(f"New best model saved! Reward: {best_reward:.1f}")
                    
                    # Save checkpoint every 10000 steps
                    if i % 10000 == 0:
                        model_path = f"models/model_step_{i}.pth"
                        agent.save_model(model_path)
                        print(f"Checkpoint saved: {model_path}")

        # Save final model
        agent.save_model("models/model_final.pth")
        print("Training complete! Final model saved.")