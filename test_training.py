import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

if __name__ == "__main__":
	# multiprocess environment
	# for now, it doens't make sense to have multiple environment
	n_cpu = 1
	env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])

	# model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./tf_logs')
	# for i in range(3):
	# 	model.learn(total_timesteps=25000, reset_num_timesteps = False)
	# 	model.save("ppo2_cartpole")

	# del model # remove to demonstrate saving and loading

	model = PPO2.load("ppo2_cartpole")

	# Enjoy trained agent
	obs = env.reset()
	while True:
	    action, _states = model.predict(obs)
	    obs, rewards, dones, info = env.step(action)
	    env.render()