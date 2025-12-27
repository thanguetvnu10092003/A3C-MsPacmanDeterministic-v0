import gymnasium as gym
from Environment.Preprocess import PreprocessAtari
import ale_py
gym.register_envs(ale_py)

def make_env(mode):
    # MsPacmanDeterministic-v0 đã được đổi tên trong Gymnasium mới
    # Sử dụng ALE/MsPacman-v5 với frameskip=4 (tương đương Deterministic)
    env = gym.make('ALE/MsPacman-v5', render_mode=mode, frameskip=4)
    env = PreprocessAtari(env, height=42, width=42, crop=lambda img: img, dim_order='pytorch', color=False, n_frames=20)
    return env
