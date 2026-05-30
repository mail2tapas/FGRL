
from workflows.dag_generator import load_workflow_dataset
from env.fog_env import FogSchedulingEnv
from rl.ppo_agent import create_agent
import config

graph = load_workflow_dataset(
    'datasets/montage/workflow_tasks.csv',
    'datasets/montage/dag_dependencies.csv'
)

env = FogSchedulingEnv(graph)
model = create_agent(env)
model.learn(total_timesteps=config.TOTAL_TIMESTEPS)
model.save(config.MODEL_PATH)

print('Training completed')
