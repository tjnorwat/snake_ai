from Snake import Snake

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.algorithm import Algorithm
from ray import tune 

# ray.init()
# checkpoint_path = 'models/1667202595/checkpoint_000001/checkpoint-1' 
# checkpoint_path = 'snake_V1/PPO/PPO_Snake_04b80_00000_0_2022-10-31_01-34-06/checkpoint_000001/checkpoint-1' # using lstm
# checkpoint_path = 'snake_V1/PPO/PPO_Snake_7a6de_00000_0_2022-10-31_01-44-33/checkpoint_000001/checkpoint-1' # not using lstm
# checkpoint_path = 'snake_V1/PPO/PPO_Snake_84181_00000_0_2022-10-31_11-46-07/checkpoint_000001/checkpoint-1' # not using lstm
# config = ppo.PPOConfig().framework('torch').training(model=dict(use_lstm=True,),).resources(num_gpus=1)


# config = ppo.PPOConfig().framework('torch')

# algo = config.build(env=Snake)
# # algo.restore(checkpoint_path=checkpoint_path)
# algo.load_checkpoint(checkpoint_path=checkpoint_path)

# algo = tune.Tuner().restore(checkpoint_path=checkpoint_path)

# agent = ppo.PPO(env=Snake, config={'framework': 'torch'})
# agent = ppo.PPO(env=Snake, config={'framework': 'torch'})
# agent.restore(checkpoint_path=checkpoint_path)
# algo = Algorithm.load_checkpoint(checkpoint_path=checkpoint_path)
# algo = ppo.PPO.load_checkpoint(checkpoint_path=checkpoint_path)


config2 = {
    'env': Snake,
    'env_config': {},
    'framework': 'torch',
    'gamma': .99,
    'lambda': .95,
    'lr': 0.0003,
    'vf_loss_coeff': .5,
    'clip_param': .2,
    'sgd_minibatch_size': 4096,
    'train_batch_size': 32768,
    'num_workers': 8,
    'num_gpus': 1,
    'grad_clip': .5,
    'model': {
        'use_lstm': True
    },
    'batch_mode': 'complete_episodes',
    'evaluation_num_workers': 1,
    'evaluation_config': {
        'render_env': True
    }
}
env = Snake(size=6, time_between_moves=100)
obs = env.reset()


checkpoint_path = 'models/checkpoint_000001/checkpoint-1'

agent = ppo.PPO(config=config2)
agent.load_checkpoint(checkpoint_path)

print(agent.get_policy().export_model('exported'))
agent.restore()
# print(agent.get_config())

# agent.evaluate()

# while True:
#     action = agent.compute_single_action(obs)
#     obs, rewards, dones, info = env.step(action=action)
#     env.render(renderer=100)
#     if dones:
#         obs = env.reset()
#         env.render()