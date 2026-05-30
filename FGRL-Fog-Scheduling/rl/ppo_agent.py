
from stable_baselines3 import PPO
import config

def create_agent(env):
    return PPO(
        'MlpPolicy',
        env,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        verbose=1
    )
