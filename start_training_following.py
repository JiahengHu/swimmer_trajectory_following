import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
import mujoco_py
from snake_env.gym_swimmer_env_baseline import SwimmerLocomotionEnv

# multiprocess environment
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
fixed_path = [(-0.2*i, 0) for i in range(30)]
use_random_path = True
robot_k = 1.0
robot_link_length = 0.3
gamma = 0.995

resume = 1


if __name__ == "__main__":
	# multiprocess environment
	# for now, it doens't make sense to have multiple environment
	n_cpu = 8
	env = SubprocVecEnv([lambda: SwimmerLocomotionEnv(
			path = fixed_path, 
			random_path = use_random_path, 
	        use_hard_path = False, 
	        robot_link_length = robot_link_length) for i in range(n_cpu)])
	if resume:
		model = PPO2.load("model/traj_follow/ppo_weight_99", env = env, verbose=1, tensorboard_log='./tf_logs/traj_follow/')
	else:
		model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./tf_logs')
	model.save("model/traj_follow_curve/ppo_weight_"+str(0))
	for i in range(100):
		model.learn(total_timesteps=250000, reset_num_timesteps = False)
		model.save("model/traj_follow_curve/ppo_weight_"+str(i))