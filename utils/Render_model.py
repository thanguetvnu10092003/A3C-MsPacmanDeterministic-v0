import os
import numpy as np


def render_agent_performance(agent, env, max_steps=3000):
    """
    Render the agent playing the game.
    
    Args:
        agent: Trained agent
        env: Environment with render_mode='human'
        max_steps: Maximum steps per episode
    """
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < max_steps:
        env.render()
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action[0])
        total_reward += reward
        steps += 1

    env.close()
    print(f"Total reward: {total_reward} | Steps: {steps}")


def record_agent_performance(agent, env, output_path="recordings/gameplay.gif", 
                              max_steps=2000, fps=30):
    """
    Record the agent playing the game and save as GIF or MP4.
    
    Args:
        agent: Trained agent
        env: Environment with render_mode='rgb_array'  
        output_path: Path to save the recording (.gif or .mp4)
        max_steps: Maximum steps to record
        fps: Frames per second for the output
    
    Returns:
        Path to the saved recording
    """
    try:
        import imageio
    except ImportError:
        print("Please install imageio: pip install imageio imageio-ffmpeg")
        return None
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    frames = []
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print(f"Recording gameplay (max {max_steps} steps)...")
    
    while not done and steps < max_steps:
        # Capture frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Agent action
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action[0])
        total_reward += reward
        steps += 1
        
        if steps % 500 == 0:
            print(f"  Step {steps}, Reward so far: {total_reward}")

    env.close()
    
    print(f"Captured {len(frames)} frames")
    print(f"Total reward: {total_reward} | Steps: {steps}")
    
    if len(frames) == 0:
        print("No frames captured!")
        return None
    
    # Save as GIF or MP4
    print(f"Saving to {output_path}...")
    
    if output_path.endswith('.gif'):
        # For GIF, reduce fps to keep file size manageable
        gif_fps = min(fps, 15)
        # Skip frames to reduce size
        skip = max(1, len(frames) // 300)  # Max ~300 frames for GIF
        frames_to_save = frames[::skip]
        imageio.mimsave(output_path, frames_to_save, fps=gif_fps, loop=0)
    else:
        # For MP4
        imageio.mimsave(output_path, frames, fps=fps)
    
    print(f"Saved recording to: {output_path}")
    return output_path
