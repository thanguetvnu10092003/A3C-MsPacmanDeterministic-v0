"""
Stable-Baselines3 Training Script for Ms. Pac-Man
Compare PPO, A2C, DQN algorithms with custom A3C implementation
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)


def make_atari_env(render_mode='rgb_array'):
    """Create and wrap Atari environment with EpisodicLifeEnv (standard training)"""
    env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
    env = AtariWrapper(env)
    return env


def make_atari_env_full_game(render_mode='rgb_array'):
    """
    Create Atari environment WITHOUT EpisodicLifeEnv
    This trains the agent to handle all 3 lives in one episode,
    which helps it learn to recover from deaths better.
    """
    from stable_baselines3.common.atari_wrappers import (
        NoopResetEnv, MaxAndSkipEnv, FireResetEnv, ClipRewardEnv, WarpFrame
    )
    env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    # NO EpisodicLifeEnv - agent learns to handle all lives!
    env = WarpFrame(env)  # Resize to 84x84 grayscale
    env = ClipRewardEnv(env)
    return env


def train_sb3(algorithm='PPO', total_timesteps=100000, n_envs=8, save_dir='models_sb3', full_game=False):
    """
    Train using Stable-Baselines3
    
    Args:
        algorithm: 'PPO', 'A2C', or 'DQN'
        total_timesteps: Total training steps (recommend 1M+ for good performance)
        n_envs: Number of parallel environments
        save_dir: Base directory to save models
        full_game: If True, train on full games (all lives) instead of per-life episodes
    """
    # Create algorithm-specific directory (all files go here)
    algo_save_dir = os.path.join(save_dir, algorithm.upper())
    os.makedirs(algo_save_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Training {algorithm} on Ms. Pac-Man")
    print(f"{'='*50}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Full game mode: {full_game} {'(trains all 3 lives per episode)' if full_game else '(per-life episodes)'}")
    print(f"Save directory: {algo_save_dir}")
    print(f"{'='*50}\n")
    
    # Create vectorized environment
    if full_game:
        env = DummyVecEnv([lambda: make_atari_env_full_game() for _ in range(n_envs)])
    else:
        env = DummyVecEnv([lambda: make_atari_env() for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)
    
    # Create eval environment
    eval_env = DummyVecEnv([lambda: make_atari_env()])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    
    # Callbacks - everything saves to algo_save_dir
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // n_envs,
        save_path=algo_save_dir,
        name_prefix="checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=algo_save_dir,
        log_path=algo_save_dir,  # Evaluation logs also go here
        eval_freq=5000 // n_envs,
        n_eval_episodes=5,
        deterministic=True
    )
    
    # Select algorithm
    if algorithm.upper() == 'PPO':
        model = PPO(
            'CnnPolicy',
            env,
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=algo_save_dir
        )
    elif algorithm.upper() == 'A2C':
        model = A2C(
            'CnnPolicy',
            env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=algo_save_dir
        )
    elif algorithm.upper() == 'DQN':
        model = DQN(
            'CnnPolicy',
            env,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            verbose=1,
            tensorboard_log=algo_save_dir
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    final_path = os.path.join(algo_save_dir, 'final_model')
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model


def evaluate_sb3(algorithm='PPO', model_path=None, n_episodes=10, render=False):
    """
    Evaluate a trained SB3 model
    
    Args:
        algorithm: 'PPO' or 'A2C'
        model_path: Path to saved model (without .zip extension)
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
    """
    # Default model path - now uses algorithm-specific folder
    if model_path is None:
        # Try best_model first, then final_model
        best_path = f'models_sb3/{algorithm.upper()}/best_model'
        final_path = f'models_sb3/{algorithm.upper()}/final_model'
        if os.path.exists(best_path + '.zip'):
            model_path = best_path
        else:
            model_path = final_path
    
    if not os.path.exists(model_path + '.zip'):
        print(f"Model not found: {model_path}.zip")
        print("Please train first with: python train_sb3.py --mode train")
        return
    
    print(f"\n{'='*50}")
    print(f"Evaluating {algorithm} on Ms. Pac-Man")
    print(f"{'='*50}\n")
    
    # Load model
    if algorithm.upper() == 'PPO':
        model = PPO.load(model_path)
    elif algorithm.upper() == 'A2C':
        model = A2C.load(model_path)
    else:
        model = DQN.load(model_path)
    
    # Create environment
    render_mode = 'human' if render else 'rgb_array'
    env = DummyVecEnv([lambda: make_atari_env(render_mode)])
    env = VecFrameStack(env, n_stack=4)
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    return mean_reward, std_reward


def demo_sb3(algorithm='PPO', model_path=None):
    """
    Demo mode - runs continuously until Ctrl+C
    Similar to original A3C demo
    """
    # Default model path - now uses algorithm-specific folder
    if model_path is None:
        best_path = f'models_sb3/{algorithm.upper()}/best_model'
        final_path = f'models_sb3/{algorithm.upper()}/final_model'
        if os.path.exists(best_path + '.zip'):
            model_path = best_path
        else:
            model_path = final_path
    
    if not os.path.exists(model_path + '.zip'):
        print(f"Model not found: {model_path}.zip")
        print("Please train first with: python train_sb3.py --mode train")
        return
    
    print(f"\n{'='*50}")
    print(f"Demo: {algorithm} playing Ms. Pac-Man")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*50}\n")
    
    # Load model
    if algorithm.upper() == 'PPO':
        model = PPO.load(model_path)
    elif algorithm.upper() == 'A2C':
        model = A2C.load(model_path)
    else:
        model = DQN.load(model_path)
    
    # Create environment with human rendering
    env = DummyVecEnv([lambda: make_atari_env('human')])
    env = VecFrameStack(env, n_stack=4)
    
    episode = 0
    total_rewards = []
    
    try:
        while True:
            episode += 1
            obs = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                steps += 1
                done = done[0]
            
            total_rewards.append(episode_reward)
            avg_reward = np.mean(total_rewards)
            print(f"Episode {episode}: Reward = {episode_reward:.0f} | Avg = {avg_reward:.1f} | Steps = {steps}")
            
    except KeyboardInterrupt:
        print(f"\n\n{'='*50}")
        print(f"Demo stopped after {episode} episodes")
        print(f"Average reward: {np.mean(total_rewards):.1f}")
        print(f"Best reward: {np.max(total_rewards):.0f}")
        print(f"{'='*50}")
    
    env.close()


def record_sb3(algorithm='PPO', model_path=None, output_path='recordings/sb3_gameplay.gif', max_steps=10000):
    """
    Record gameplay from SB3 model until episode ends (game over)
    
    Args:
        algorithm: PPO, A2C, or DQN
        model_path: Path to model (auto-detect if None)
        output_path: Where to save GIF
        max_steps: Safety limit to prevent infinite loops
    """
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio")
        return
    
    # Default model path - now uses algorithm-specific folder
    if model_path is None:
        best_path = f'models_sb3/{algorithm.upper()}/best_model'
        final_path = f'models_sb3/{algorithm.upper()}/final_model'
        if os.path.exists(best_path + '.zip'):
            model_path = best_path
        else:
            model_path = final_path
    
    if not os.path.exists(model_path + '.zip'):
        print(f"Model not found: {model_path}.zip")
        return
    
    print(f"Recording {algorithm} gameplay (until game over)...")
    
    # Load model
    if algorithm.upper() == 'PPO':
        model = PPO.load(model_path)
    elif algorithm.upper() == 'A2C':
        model = A2C.load(model_path)
    else:
        model = DQN.load(model_path)
    
    # Create environment with AtariWrapper but terminal_on_life_loss=False
    # This keeps the preprocessing but doesn't end episode on life loss
    import gymnasium as gym
    import ale_py
    from stable_baselines3.common.atari_wrappers import (
        NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, 
        ClipRewardEnv, WarpFrame
    )
    gym.register_envs(ale_py)
    
    # Create env with manual wrappers (like AtariWrapper but without EpisodicLifeEnv)
    raw_env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
    raw_env = NoopResetEnv(raw_env, noop_max=30)
    raw_env = MaxAndSkipEnv(raw_env, skip=4)
    # NO EpisodicLifeEnv - we want all lives!
    raw_env = WarpFrame(raw_env)  # Resize to 84x84 grayscale
    raw_env = ClipRewardEnv(raw_env)
    
    env = DummyVecEnv([lambda: raw_env])
    env = VecFrameStack(env, n_stack=4)
    
    frames = []
    obs = env.reset()
    total_reward = 0
    step = 0
    game_over = False
    lives = 3  # Ms. Pac-Man starts with 3 lives
    
    # Run until all lives lost (game over) or max_steps reached
    while not game_over and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Get frame from unwrapped env for proper rendering
        frame = env.envs[0].unwrapped.render()
        if frame is not None:
            frames.append(frame)
        
        total_reward += reward[0]
        step += 1
        
        # Check lives from info
        if 'lives' in info[0]:
            current_lives = info[0]['lives']
            if current_lives < lives:
                print(f"  Lost a life! Lives remaining: {current_lives}")
                lives = current_lives
            if current_lives == 0:
                game_over = True
        elif done[0]:
            game_over = True
        
        if step % 500 == 0:
            print(f"  Step {step}, Reward: {total_reward:.0f}, Lives: {lives}")
    
    env.close()
    
    print(f"Episode finished! Steps: {step}, Total reward: {total_reward:.0f}")
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    if len(frames) > 0:
        skip = max(1, len(frames) // 300)
        imageio.mimsave(output_path, frames[::skip], fps=15, loop=0)
        print(f"Saved to: {output_path}")
        print(f"Total reward: {total_reward:.0f}")
    else:
        print("No frames captured!")


def main():
    parser = argparse.ArgumentParser(description='Stable-Baselines3 Training for Ms. Pac-Man')
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'eval', 'record', 'demo'],
                        help='Mode: train, eval, record, or demo')
    parser.add_argument('--algorithm', type=str, default='PPO', 
                        choices=['PPO', 'A2C', 'DQN'],
                        help='Algorithm to use')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--envs', type=int, default=8,
                        help='Number of parallel environments')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='recordings/sb3_gameplay.gif',
                        help='Output path for recording')
    parser.add_argument('--full-game', action='store_true',
                        help='Train on full games (all 3 lives) - helps agent learn to recover from deaths')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_sb3(args.algorithm, args.timesteps, args.envs, full_game=args.full_game)
    elif args.mode == 'eval':
        evaluate_sb3(args.algorithm, n_episodes=args.episodes)
    elif args.mode == 'demo':
        demo_sb3(args.algorithm)
    elif args.mode == 'record':
        record_sb3(args.algorithm, output_path=args.output)


if __name__ == '__main__':
    main()
