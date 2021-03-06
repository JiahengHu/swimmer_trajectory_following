from stable_baselines import TRPO
from stable_baselines import PPO2
from snake_env.reynolds_swimmer_env import SwimmerLocomotionEnv
import numpy as np
fixed_path = [(-0.2*i, 0) for i in range(30)]

use_random_path = True
robot_k = 1.0
robot_link_length = 0.3

#these are for testing
#model = TRPO.load("trpo_swimmer")
if(use_random_path):
    model = PPO2.load("model/traj_follow_curve/reynolds_ppo_weight_22")
else:
    model = PPO2.load("model/traj_follow_line/ppo_weight_99")
env = SwimmerLocomotionEnv(
		path = fixed_path, 
		random_path = use_random_path, 
        use_hard_path = False, 
        robot_link_length = robot_link_length,
        robot_k = robot_k,
        record_trajectory = True)

obs = env.reset()
total_reward = 0
x_list = []
for i in range(10000):
    action, _states = model.predict(obs)
    #step_time = 0.5
    #action = [-0.8*np.sin(step_time*i), 0.8*np.cos(step_time*i)]
    # print("start of step")
    #print(action)
    x_list.append(action[1])
    obs, rewards, dones, info = env.step(action)
    # print(obs)
    # print(rewards)
    total_reward+=rewards
    if((i+1)%1000==0):
    	env.render(save = False, save_id = i//10)
     	#pass
    	
    if(dones):
    	break
print(total_reward)
import matplotlib.pyplot as plt
lines = plt.plot(range(1000), x_list)
lab = plt.xlabel('time')
leg = plt.legend('u1')
plt.show()
env.draw_trajectory()