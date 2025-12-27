import os


def load_model(agent, models_dir="models", episode=None, step=None, best=False):
    """
    Load a trained model.
    
    Args:
        agent: Agent instance to load weights into
        models_dir: Directory containing saved models
        episode: Load model from specific episode (legacy format)
        step: Load model from specific training step (new format)
        best: If True, load the best performing model
    
    Returns:
        Agent with loaded weights, or None if not found
    """
    if best:
        model_path = os.path.join(models_dir, "model_best.pth")
    elif step is not None:
        model_path = os.path.join(models_dir, f"model_step_{step}.pth")
    elif episode is not None:
        model_path = os.path.join(models_dir, f"model_episode_{episode}.pth")
    else:
        # Try to load final model
        model_path = os.path.join(models_dir, "model_final.pth")
    
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        agent.load_model(model_path)
        return agent
    else:
        print(f"Model not found: {model_path}")
        return None
