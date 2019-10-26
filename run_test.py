from stable_baselines import PPO2
from snake_env.gym_swimmer_env import SwimmerLocomotionEnv
import numpy as np
fixed_path = [(-0.2*i,-0.2*i) for i in range(30)]
#fixed_path = [(0,-0.2*i) for i in range(30)]
#fixed_path = [(-0.2*i,0.2/6*i) for i in range(30)]

use_random_path = False
robot_k = 1.0
robot_link_length = 0.3

#these are for testing
model = PPO2.load("ppo2_swimmer")
env = SwimmerLocomotionEnv(
		path = fixed_path, 
		random_path = use_random_path, 
        use_hard_path = False, 
        robot_link_length = robot_link_length,
        robot_k = robot_k,
        record_trajectory = True)

obs = env.reset()
total_reward = 0
for i in range(10000):
    step_time = 0.5
    #action = [-0.8*np.sin(step_time*i), 0.8*np.cos(step_time*i)]
    action, _states = model.predict(obs)
    
    # print("start of step")
    # print(action)
    obs, rewards, dones, info = env.step(action)
    # print(obs)
    # print(rewards)
    total_reward+=rewards
    if(i%10==0):
    	env.render()
    	pass
    	

    if(dones):
    	break
print(total_reward)
env.draw_trajectory()