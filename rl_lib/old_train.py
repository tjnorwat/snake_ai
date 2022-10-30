import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.ppo import PPO

from Snake import Snake, Actions

parameter_search_config = {
    "env": Snake,
    "framework": "torch",

    # Hyperparameter tuning
    "model": {
      "fcnet_hiddens": ray.tune.grid_search([[256], [256]]),
      "fcnet_activation": ray.tune.grid_search(["linear", "relu"]),
      'use_lstm': True
    },
}

# To explicitly stop or restart Ray, use the shutdown API.
# ray.shutdown()
# stop = {"episode_reward_mean": 195}


# ray.init(
#   num_cpus=8,
#   include_dashboard=False,
#   ignore_reinit_error=True,
#   log_to_driver=False,
# )

# parameter_search_analysis = ray.tune.run(
#   "PPO",
#   config=parameter_search_config,
#   stop=stop,
#   num_samples=5,
#   metric="timesteps_total",
#   mode="min",
# )



config = {
          
}

algo = PPO(env=Snake, config=parameter_search_config)

while True:
  print(algo.train())



