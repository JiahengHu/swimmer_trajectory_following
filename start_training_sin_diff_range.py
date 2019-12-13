from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from snake_env.gym_swimmer_forward_sin import SwimmerLocomotionEnv
import sys

# multiprocess environment
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
fixed_path = [(-0.2*i,-0.2*i) for i in range(30)]
use_random_path = False
robot_k = 1.0
robot_link_length = 0.3
gamma = 0.995




if __name__ == "__main__":
	resume = False
	if(len(sys.argv)>1):
		resume = int(sys.argv[1])
		print(f"resume is: {resume}")
	else:
		print("no system argument")

	max_vel = 0.6
	for j in range(1):
		action_range = max_vel/(j+5)
		n_cpu = 6
		env = SubprocVecEnv([lambda: SwimmerLocomotionEnv(
				path = fixed_path, 
				random_path = use_random_path, 
				action_range = action_range,
		        use_hard_path = False, 
		        robot_link_length = robot_link_length) for i in range(n_cpu)])

		print("new env")
		#two layers of size 64
		model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./tf_logs/test_action_range/')


		for i in range(20):
			model.learn(total_timesteps=250000, reset_num_timesteps = False)
			model.save("ppo2_swimmer_action_range_"+str(action_range))

		del env