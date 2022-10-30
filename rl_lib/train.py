import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.ppo import PPO

from Snake import Snake, Actions


config = {
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
    'batch_mode': 'complete_episodes'
}

ray.shutdown()
ray.init()
algo = PPO(env=Snake, config={'framework': 'torch'})

while True:
    print(algo.train())