import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import TRPO
import mujoco_py

import pybullet
import pybullet_data
import pybullet_envs

if __name__ == "__main__":
	# multiprocess environment
	# for now, it doens't make sense to have multiple environment
	n_cpu = 1
	env = SubprocVecEnv([lambda: gym.make('Swimmer-v2') for i in range(n_cpu)])
	#model = PPO2.load("ppo2_swimmer", env = env, verbose=1, tensorboard_log='./tf_logs/hopper')
	#model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./tf_logs')
	
	# for i in range(1000):
	# 	model.learn(total_timesteps=25000, reset_num_timesteps = False)
	# 	model.save("ppo2_swimmer")

	# del model # remove to demonstrate saving and loading

	#model = TRPO.load("model/gym_swimmer/ppo2_swimmer_test_gym_step30")
	#model = TRPO.load("model/gym_swimmer/ppo2_swimmer_test_gym_step83")
	model = TRPO.load("model/prev_policy/ppo2_swimmer_trpo")

	# Enjoy trained agent
	obs = env.reset()
	while True:
	    action, _states = model.predict(obs)
	    print(action)
	    obs, rewards, dones, info = env.step(action)
	    print(dones)
	    env.render()