
from stable_baselines3 import PPO
from workflows.dag_generator import load_workflow_dataset
from env.fog_env import FogSchedulingEnv

graph = load_workflow_dataset(
    'datasets/montage/workflow_tasks.csv',
    'datasets/montage/dag_dependencies.csv'
)

env = FogSchedulingEnv(graph)

model = PPO.load('checkpoints/ppo_scheduler.zip')

obs,_ = env.reset()

done=False
reward_sum=0

while not done:
    action,_ = model.predict(obs)
    obs,reward,done,truncated,info = env.step(action)
    reward_sum += reward

print('Total Reward:', reward_sum)
