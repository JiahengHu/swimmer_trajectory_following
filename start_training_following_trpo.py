import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import TRPO
import mujoco_py
from snake_env.gym_swimmer_env import SwimmerLocomotionEnv

# multiprocess environment
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
fixed_path = [(-0.2*i, 0) for i in range(30)]
use_random_path = False
robot_k = 1.0
robot_link_length = 0.3
gamma = 0.995



if __name__ == "__main__":
	# multiprocess environment
	# for now, it doens't make sense to have multiple environment
	n_cpu = 1
	env = SubprocVecEnv([lambda: SwimmerLocomotionEnv(
			path = fixed_path, 
			random_path = use_random_path, 
	        use_hard_path = False, 
	        robot_link_length = robot_link_length) for i in range(n_cpu)])
	#model = PPO2.load("ppo2_hopper", env = env, verbose=1, tensorboard_log='./tf_logs/hopper')
	model = TRPO(MlpPolicy, env, verbose=1, tensorboard_log='./tf_logs')
	
	for i in range(100):
		model.learn(total_timesteps=250000, reset_num_timesteps = False)
		model.save("real_trpo_swimmer_traj_following")
