from stable_baselines.ddpg import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import DDPG
from snake_env.gym_swimmer_forward_vel import SwimmerLocomotionEnv
import sys
import numpy as np

# multiprocess environment
# n_cpu = 4
# env = SubprocVecEnv([lambda: gym.make('CartPole-v1') for i in range(n_cpu)])
fixed_path = [(-0.2*i,-0.2*i) for i in range(30)]
use_random_path = False
robot_k = 1.0
robot_link_length = 0.3
pre_train = True
collect_num = 10


if __name__ == "__main__":
	resume = False
	if(len(sys.argv)>1):
		resume = int(sys.argv[1])
		print(f"resume is: {resume}")
	else:
		print("no system argument")

	n_cpu = 1
	# env = SubprocVecEnv([lambda: SwimmerLocomotionEnv(
	# 		path = fixed_path, 
	# 		random_path = use_random_path, 
	#         use_hard_path = False, 
	#         robot_link_length = robot_link_length,
	#         robot_k = robot_k) for i in range(n_cpu)])
	env = SwimmerLocomotionEnv(
			path = fixed_path, 
			random_path = use_random_path, 
	        use_hard_path = False, 
	        robot_link_length = robot_link_length,
	        record_trajectory = True,
	        robot_k = robot_k)

	test_env = SwimmerLocomotionEnv(
		path = fixed_path, 
		random_path = use_random_path, 
        use_hard_path = False, 
        robot_link_length = robot_link_length,
        record_trajectory = True,
        robot_k = robot_k)

	if resume:
		print("resuming training")
		model = DDPG.load("DDPG_swimmer", env = env, verbose=1, tensorboard_log='./tf_logs/swimmer')
	else:
		print("not resuming")
		#two layers of size 64
		model = DDPG(MlpPolicy, env, verbose=1, buffer_size = 200000, nb_eval_steps = 1000, eval_env = test_env, demo_buffer = True, tensorboard_log='./tf_logs/swimmer')




	#the callback function
	best_mean_reward, n_steps = -np.inf, 0
	log_dir = "model/ddpg/"

	def callback(_locals, _globals):
	    """
	    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
	    :param _locals: (dict)
	    :param _globals: (dict)
	    """
	    global n_steps, best_mean_reward
	    # Print stats every 1000 calls
	    if (n_steps + 1) % 10000 == 0:
	        # Evaluate policy training performance
	        #x, y = ts2xy(load_results(log_dir), 'timesteps')
	        # if len(x) > 0:
	        #     mean_reward = np.mean(y[-100:])
	        #     print(x[-1], 'timesteps')
	        #     print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

	            # # New best model, you could save the agent here
	            # if mean_reward > best_mean_reward:
	            #     best_mean_reward = mean_reward
	            #     # Example for saving best model
	            #     print("Saving new best model")
	        _locals['self'].save(log_dir + 'best_model.pkl')
	    n_steps += 1
	    # Returning False will stop training early
	    return True

	freq = None
	amp = None
	def collect_buffer(obs, done):
		global freq,amp
		if done:
			#initialize the map for parameters
			#freq = np.random.rand()*1.5 + 0.5 #0.5-2
			freq = 1
			amp = 0.7 #np.random.rand()*0.7 + 0.3  #0.3-1
			#print(f"frequency:{freq}, amplitude:{amp}")
		action = np.asarray([np.cos(obs[-1]*freq), np.sin(obs[-1]*freq)]) * amp
		return action


	for i in range(1):
		model.learn(total_timesteps=2500000, reset_num_timesteps = False, callback = callback, 
			pre_collect = 1000, pre_collect_function = collect_buffer)
		model.save("DDPG_swimmer")

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