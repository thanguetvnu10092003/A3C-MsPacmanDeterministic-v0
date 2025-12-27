from utils.Training import TrainingModel
from utils.Make_Env import make_env
from Network.Agent import Agent
from utils.Render_model import render_agent_performance
from utils.Load_model import load_model
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='A3C Ms. Pac-Man Training and Evaluation')
    parser.add_argument('--mode', type=str, default='demo', choices=['train', 'demo'],
                        help='Mode: train (training) or demo (run saved model)')
    parser.add_argument('--epochs', type=int, default=100001,
                        help='Number of training epochs')
    parser.add_argument('--envs', type=int, default=8,
                        help='Number of parallel environments (8 recommended for 4GB GPU)')
    parser.add_argument('--load-step', type=int, default=None,
                        help='Load model from specific training step')
    parser.add_argument('--load-best', action='store_true',
                        help='Load the best performing model')
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    # Create environment
    env = make_env('rgb_array')
    state_shape = env.observation_space.shape
    number_actions = env.action_space.n
    
    print('=' * 50)
    print('A3C Ms. Pac-Man - Upgraded Model')
    print('=' * 50)
    print(f'Observation shape: {state_shape}')
    print(f'Number of actions: {number_actions}')
    print(f'Actions: {env.unwrapped.get_action_meanings()}')
    print('=' * 50)

    # Create agent with correct input channels
    input_channels = state_shape[0]  # Number of stacked frames
    agent = Agent(number_actions, input_channels=input_channels)
    
    if args.mode == 'train':
        print(f'\nStarting training for {args.epochs} epochs with {args.envs} parallel environments...')
        print('This will take a while. Press Ctrl+C to stop.\n')
        
        training_model = TrainingModel(env, training_epochs=args.epochs, number_of_envs=args.envs)
        training_model.train(agent)
        
    elif args.mode == 'demo':
        # Load model
        if args.load_best:
            loaded_agent = load_model(agent, best=True)
        elif args.load_step:
            loaded_agent = load_model(agent, step=args.load_step)
        else:
            # Try to load best model by default
            loaded_agent = load_model(agent, best=True)
            if loaded_agent is None:
                loaded_agent = load_model(agent)  # Try final model
        
        if loaded_agent is None:
            print("No trained model found! Please train first with: python main.py --mode train")
            return
            
        render_env = make_env('human')
        render_agent_performance(loaded_agent, render_env)


if __name__ == '__main__':
    main()
