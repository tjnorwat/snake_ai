from tabnanny import check
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.ppo import PPO
import time
from Snake import Snake, Actions
from ray.tune.logger import TBXLoggerCallback
from ray import tune, air
import ray.rllib.algorithms.ppo as ppo

models_dir = f'models/{int(time.time())}/'

config1 = {
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
    'batch_mode': 'complete_episodes'
}

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

config3 = {
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
    'batch_mode': 'complete_episodes'
}

# ray.shutdown()
# need 1 more cpus to workers 
ray.init(
    num_cpus=9,
    include_dashboard=True
)

# print(ray.get_webui_url())
# algo = PPO(env=Snake, config={'env_config': {}, 'framework': 'torch'})
# algo = PPO(env=Snake, config=config, callbacks=TBXLoggerCallback())

# tune.Tuner().restore





# tune.run('PPO', config={
#     'env': Snake,
#     'framework': 'torch',
#     'num_workers': 8,
#     'num_envs_per_worker': 12,
#     'num_gpus': 1,
# }, local_dir='snake_V1',
# checkpoint_freq=1)


# tune.run('PPO', config={
#     'env': Snake,
#     'framework': 'torch',
#     'gamma': .99,
#     'lambda': .95,
#     'lr': 0.0003,
#     'vf_loss_coeff': .5,
#     'clip_param': .2,
#     'sgd_minibatch_size': 4096,
#     'train_batch_size': 32768,
#     'num_workers': 8,
#     'num_envs_per_worker': 12,
#     'num_gpus': 1,
#     'grad_clip': .5,
#     'batch_mode': 'complete_episodes',
#     'model': {
#         'use_lstm': True
#     },
#     'horizon': 15_000
# }, local_dir='snake_V1',
# checkpoint_freq=1,
# verbose=0
# )


# algo = ppo.PPO(
#     env=Snake,
#     config=config1
# )

# algo.train()
# algo.save(models_dir)



# tune.run
# algo = ppo.PPO(env=MyEnv, config={"env_config": {}, })

# while True:
#     print(algo.train())
#     algo.save(models_dir)




# algo = PPO(config=config2)
# algo.train()
# algo.evaluate()
# algo.save('models')
# algo.get_config()


analysis = tune.Tuner(
    'PPO',
    run_config=air.RunConfig(
        stop={'timesteps_total': 1000},
        local_dir='saved_models',
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True,
        ),
    ),
    param_space={
        'env': Snake, 
        'lr': 1e-3, 
        'framework': 'torch', 
        'num_gpus': 1, 
        'model': {'use_lstm': True} 
    }
).fit()

print('finished training')
# analysis.default_metric = 'episode_reward_mean'
# analysis.default_mode = 'max'
checkpoint_path = analysis.get_best_result().checkpoint
# print('CHECKPOINT PATH IS ', checkpoint_path)

agent = PPO(env=Snake, config={'framework': 'torch', 'model': {'use_lstm': True}})
agent.restore(checkpoint_path=checkpoint_path)

state = agent.get_policy().get_initial_state()
# print(state)

env = Snake()
obs = env.reset()
while True:
    action, state, fetch_dict = agent.compute_single_action(observation=obs, state=state)
    print('THIS IS THE ACTION', action)
    obs, rewards, dones, info = env.step(action)
    env.render(renderer=100)
    if dones:
        obs = env.reset()
        env.render()