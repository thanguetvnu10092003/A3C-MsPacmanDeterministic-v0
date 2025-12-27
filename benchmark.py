"""
Benchmark and Compare Different RL Algorithms
Compares: Custom A3C vs Stable-Baselines3 (PPO, A2C, DQN)
"""

import os
import sys
import argparse
import numpy as np
import json
from datetime import datetime


def evaluate_custom_a3c(n_episodes=10):
    """Evaluate custom A3C implementation"""
    from utils.Make_Env import make_env
    from Network.Agent import Agent
    from utils.Load_model import load_model
    
    env = make_env('rgb_array')
    input_channels = env.observation_space.shape[0]
    number_actions = env.action_space.n
    
    agent = Agent(number_actions, input_channels=input_channels)
    loaded_agent = load_model(agent, best=True)
    
    if loaded_agent is None:
        print("Custom A3C model not found!")
        return None
    
    rewards = []
    for ep in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 10000:
            action = loaded_agent.act(state)
            state, reward, done, _, _ = env.step(action[0])
            total_reward += reward
            steps += 1
        
        rewards.append(total_reward)
        print(f"  Episode {ep+1}: {total_reward:.0f}")
    
    env.close()
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'rewards': rewards
    }


def evaluate_sb3(algorithm='PPO', n_episodes=10):
    """Evaluate Stable-Baselines3 model"""
    try:
        from stable_baselines3 import PPO, A2C, DQN
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
        from stable_baselines3.common.atari_wrappers import AtariWrapper
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
    except ImportError:
        print("Stable-Baselines3 not installed! Run: pip install stable-baselines3[extra]")
        return None
    
    model_path = f'models_sb3/{algorithm.lower()}_mspacman_final'
    
    if not os.path.exists(model_path + '.zip'):
        print(f"{algorithm} model not found!")
        return None
    
    # Load model
    if algorithm.upper() == 'PPO':
        model = PPO.load(model_path)
    elif algorithm.upper() == 'A2C':
        model = A2C.load(model_path)
    else:
        model = DQN.load(model_path)
    
    # Create environment
    def make_env():
        env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
        env = AtariWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    
    rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 10000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            done = done[0]
        
        rewards.append(total_reward)
        print(f"  Episode {ep+1}: {total_reward:.0f}")
    
    env.close()
    return {
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'rewards': rewards
    }


def run_benchmark(n_episodes=10, output_file='benchmark_results.json'):
    """Run full benchmark comparing all algorithms"""
    
    print("\n" + "="*60)
    print("🎮 Ms. Pac-Man RL Algorithm Benchmark")
    print("="*60)
    print(f"Episodes per algorithm: {n_episodes}")
    print("="*60 + "\n")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_episodes': n_episodes,
        'algorithms': {}
    }
    
    # Test Custom A3C
    print("\n📊 Evaluating Custom A3C...")
    print("-" * 40)
    a3c_results = evaluate_custom_a3c(n_episodes)
    if a3c_results:
        results['algorithms']['Custom_A3C'] = a3c_results
        print(f"Mean: {a3c_results['mean']:.1f} ± {a3c_results['std']:.1f}")
    
    # Test PPO
    print("\n📊 Evaluating Stable-Baselines3 PPO...")
    print("-" * 40)
    ppo_results = evaluate_sb3('PPO', n_episodes)
    if ppo_results:
        results['algorithms']['SB3_PPO'] = ppo_results
        print(f"Mean: {ppo_results['mean']:.1f} ± {ppo_results['std']:.1f}")
    
    # Test A2C
    print("\n📊 Evaluating Stable-Baselines3 A2C...")
    print("-" * 40)
    a2c_results = evaluate_sb3('A2C', n_episodes)
    if a2c_results:
        results['algorithms']['SB3_A2C'] = a2c_results
        print(f"Mean: {a2c_results['mean']:.1f} ± {a2c_results['std']:.1f}")
    
    # Test DQN
    print("\n📊 Evaluating Stable-Baselines3 DQN...")
    print("-" * 40)
    dqn_results = evaluate_sb3('DQN', n_episodes)
    if dqn_results:
        results['algorithms']['SB3_DQN'] = dqn_results
        print(f"Mean: {dqn_results['mean']:.1f} ± {dqn_results['std']:.1f}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("📈 COMPARISON RESULTS")
    print("="*60)
    print(f"{'Algorithm':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    
    for name, data in results['algorithms'].items():
        print(f"{name:<20} {data['mean']:>10.1f} {data['std']:>10.1f} {data['min']:>10.1f} {data['max']:>10.1f}")
    
    print("="*60)
    
    # Find best algorithm
    if results['algorithms']:
        best = max(results['algorithms'].items(), key=lambda x: x[1]['mean'])
        print(f"\n🏆 Best Algorithm: {best[0]} (Mean: {best[1]['mean']:.1f})")
    
    # Save results
    with open(output_file, 'w') as f:
        # Convert numpy values to Python types for JSON
        json_results = {
            'timestamp': results['timestamp'],
            'n_episodes': results['n_episodes'],
            'algorithms': {
                name: {
                    'mean': float(data['mean']),
                    'std': float(data['std']),
                    'min': float(data['min']),
                    'max': float(data['max']),
                    'rewards': [float(r) for r in data['rewards']]
                }
                for name, data in results['algorithms'].items()
            }
        }
        json.dump(json_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark RL Algorithms on Ms. Pac-Man')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes per algorithm')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Output file for results')
    args = parser.parse_args()
    
    run_benchmark(args.episodes, args.output)


if __name__ == '__main__':
    main()
