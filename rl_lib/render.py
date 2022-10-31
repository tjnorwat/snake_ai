from Snake import Snake

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune 

# ray.init()
# checkpoint_path = 'models/1667202595/checkpoint_000001/checkpoint-1' 
# checkpoint_path = 'snake_V1/PPO/PPO_Snake_04b80_00000_0_2022-10-31_01-34-06/checkpoint_000001/checkpoint-1' # using lstm
# checkpoint_path = 'snake_V1/PPO/PPO_Snake_7a6de_00000_0_2022-10-31_01-44-33/checkpoint_000001/checkpoint-1' # not using lstm
checkpoint_path = 'snake_V1/PPO/PPO_Snake_1f2ab_00000_0_2022-10-31_03-07-54/checkpoint_005129/checkpoint-5129' # not using lstm
# config = ppo.PPOConfig().framework('torch').training(model=dict(use_lstm=True,),).resources(num_gpus=1)


# config = ppo.PPOConfig().framework('torch')

# algo = config.build(env=Snake)
# # algo.restore(checkpoint_path=checkpoint_path)
# algo.load_checkpoint(checkpoint_path=checkpoint_path)

# algo = tune.Tuner().restore(checkpoint_path=checkpoint_path)

agent = ppo.PPO(env=Snake, config={'framework': 'torch'})
agent.restore(checkpoint_path=checkpoint_path)
# algo = Algorithm.load_checkpoint(checkpoint_path=checkpoint_path)
# algo = ppo.PPO.load_checkpoint(checkpoint_path=checkpoint_path)

env = Snake(size=6, time_between_moves=100)
obs = env.reset()
while True:
    action = agent.compute_single_action(obs)
    obs, rewards, dones, info = env.step(action=action)
    env.render(renderer=100)
    if dones:
        obs = env.reset()
        env.render()