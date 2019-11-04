from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
from snake_env.gym_snake_env import SnakeLocomotionEnv
import sys

# multiprocess environment
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
fixed_path = [(-0.2*i,-0.2*i) for i in range(30)]
use_random_path = False
robot_link_length = 0.3



if __name__ == "__main__":
	resume = True
	if(len(sys.argv)>1):
		resume = int(sys.argv[1])
		print(f"resume is: {resume}")
	else:
		print("no system argument")

	n_cpu = 1
	env = SubprocVecEnv([lambda: SnakeLocomotionEnv(
			path = fixed_path, 
			random_path = use_random_path, 
	        use_hard_path = False, 
	        robot_link_length = robot_link_length) for i in range(n_cpu)])
	if resume:
		print("resuming training")
		model = PPO2.load("ppo2_snake", env = env, verbose=1, tensorboard_log='./tf_logs')
	else:
		print("not resuming")
		#two layers of size 64
		model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./tf_logs')
	for i in range(10):
		model.learn(total_timesteps=250000, reset_num_timesteps = False)
		model.save("ppo2_snake")

	# del model # remove to demonstrate saving and loading

	# #these are for testing
	# model = PPO2.load("ppo2_swimmer")
	# env = SwimmerLocomotionEnv(
	# 		path = fixed_path, 
	# 		random_path = use_random_path, 
	#         use_hard_path = False, 
	#         robot_link_length = robot_link_length,
	#         robot_k = robot_k,
	#         record_trajectory = True)


	# # Testing purpose (should be in a seperate file)
	# obs = env.reset()
	# total_reward = 0
	# for i in range(1000):
	#     action, _states = model.predict(obs)
	#     obs, rewards, dones, info = env.step(action)
	#     total_reward+=rewards
	#     if(i%100==0):
	#     	pass
	#     	#env.render()

	#     if(dones):
	#     	break
	# print(total_reward)
	# env.draw_trajectory()