import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO
import mujoco_py

import pybullet
import pybullet_data
import pybullet_envs

if __name__ == "__main__":
	# multiprocess environment
	# for now, it doens't make sense to have multiple environment
	n_cpu = 1
	env = DummyVecEnv([lambda: gym.make('Swimmer-v2') for i in range(n_cpu)])
	#model = PPO2.load("ppo2_hopper", env = env, verbose=1, tensorboard_log='./tf_logs/hopper')
	model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log='./tf_logs')
	
	for i in range(100):
		model.learn(total_timesteps=250000, reset_num_timesteps = False)
		model.save("model/gym_swimmer/ppo2_swimmer_test_gym_step"+str(i))

	# del model # remove to demonstrate saving and loading

	#model = PPO2.load("ppo2_cartpole")

	# # Enjoy trained agent
	# obs = env.reset()
	# while True:
	#     action, _states = model.predict(obs)
	#     obs, rewards, dones, info = env.step(action)
	#     env.render()