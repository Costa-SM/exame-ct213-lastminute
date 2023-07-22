import gymnasium as gym
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# Configure the logger
logger_path = "./breakoutlog"
logger = configure(logger_path, ["stdout", "csv", "tensorboard"])

# We use parallel environments to speed-up the training process
vec_env = make_vec_env("ALE/Breakout-v5", n_envs=4)

# Custom MLP policy of two layers of size 32 each with Relu activation function
policy_kwargs = dict(
    activation_fn=th.nn.ReLU, 
    net_arch=[32, 64, 128, 32]
)

# DQN Hyperparameters
gamma = 0.95
learning_rate = 0.0005          
buffer_size = 10_000

batch_size = 16
train_freq = 1          
exploration_fraction = 0.1
exploration_initial_eps = 1.0
exploration_final_eps = 0.05


# Create the agent
model = DQN("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=0,
            learning_rate=learning_rate, 
            buffer_size=buffer_size, 
            batch_size=batch_size, 
            gamma=gamma, 
            train_freq=train_freq, 
            exploration_fraction=exploration_fraction, 
            exploration_initial_eps=exploration_initial_eps, 
            exploration_final_eps=exploration_final_eps, 
            )

# Set the new logger
model.set_logger(logger)

# Start training
model.learn(total_timesteps=10_000_000)
model.save("breakout-dqn")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
