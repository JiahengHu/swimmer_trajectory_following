import gym
from gym import spaces


from numpy import pi,cos,sin
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import os
from snake_env.snake_lib import Snake_Robot


##################################
###### define hyperparameters  ###
##################################

num_of_links = 3                        #the number of links of the snake robot
num_of_points = 3                       #the number of points to look ahead
max_angle = 1                           #the range of the joint angle
time_interval = pi                      #the length of each episode of action
end_step_num = 55                       #stop an episode after a given amount of time
dist_threshold = 3                      #the value to determine if the robot has reach the end position
path_length = 50                        #the length of the randomly generated path
use_random_state = False                #whether the robot start with a random state initially
use_random_path = True                  #whether the robot should use a random path
easy_path = True                        #whether the robot should use a easy (random) path
save_trajectory = False                 #this will help keep track of the robot's state
param_robot_link_length = 0.3           #this controls the link length of the robot


##################################
###### the snake environment   ###
##################################

class SnakeLocomotionEnv(gym.Env):

  def __init__(self, path = None, random_path = use_random_path, use_hard_path = not easy_path, 
    record_trajectory = save_trajectory, 
    robot_link_length = param_robot_link_length,
    robot_mass = 0.01, robot_k = 0.01):
        
    # This determines what kind of path we are using
    self.use_hard_path = use_hard_path
    self.use_random_path = random_path
    self.save_trajectory = record_trajectory
    self.n = num_of_links
    self.action_space = spaces.Box(low = 0.05, high = 2*max_angle, 
        shape=(1,), dtype=np.float32)
    
    #state: q1, q2, x', y', t', q2', points, up/down
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=
                    (2*num_of_links + 2*num_of_points+1,), dtype=np.float32)

    self._snake_model = Snake_Robot(num_of_links, 
      link_length = robot_link_length, mass = robot_mass, k_val = robot_k)

    #the path that the robot has to follow, an array of points
    self._path = path         
    self.reset_path()
    self.reset_state()
    self.reset_param()
    
  #motor input = -amplitude * sin(t*pi/time_interval)
  def action2motor(self, action):
    action = action[0]
    q_prev = self._state[0]
    phase = self._state[-1]
    #if move down
    if(phase == 1):
      target = max(-max_angle,q_prev - action)
    else:
      target = min(max_angle, q_prev + action)
    amplitude = (q_prev - target) / 2.0
    #print(f"target: {target}, amplitutde: {amplitude}")
    return -amplitude,pi/time_interval
  
  #this function assumes 3-link robot
  def get_initial_config(self):
    #state: q1-qn, dx,dy, u1-un
    n = self.n
    dx, dy = self.vec_back(self._state[n-1], self._state[n])
    return np.hstack([self._x, self._y, self._t] + self._state[:n-1]+ 
                   [dx, dy, self._state[n+1], 0] + self._state[n+2:2*n] )
  
  
  #this returns the vector indicating the motion of the robot
  def get_motion(self, y):
    return (y[0] - self._x, y[1] - self._y)
  
  #return the unit vector inidicating the direction
  def get_direction(self):
    dir_vec = (self._path[self._start_point_index][0] - self._x, 
      self._path[self._start_point_index][1] - self._y)
    return self.vec_normalize(dir_vec)
  
  #only for 2-d vector, normalize it
  def vec_normalize(self,vec):
    length = vec[0]*vec[0]+vec[1]*vec[1]
    length = math.sqrt(length)
    return (vec[0]/length, vec[1]/length)
    
  #returns relationship between current pos and the path
  def cal_distances_to_points(self, y):
    direction_vec = self.get_direction()
    motion_vec = self.get_motion(y)
    score = self.vec_dot(direction_vec, motion_vec)
    #print(f"direction_vec:{direction_vec}, motion_vec: {motion_vec}")
    
    #need to handle the special case when start point index is the last point?
    if(self._start_point_index == len(self._path)):
    	print("warning! Starting point is the last point")
    	print(f"total step: {self._total_step}")
    	self._start_point_index -= 1

    #need to change this completely
    #calculate the new distance to each of the five points
    dist_sum = 0
    min_dist_index = self._start_point_index - 1
    min_dist = float("inf")
    vec = (y[0]- self._path[self._start_point_index - 1][0], 
      y[1]- self._path[self._start_point_index - 1][1])
    vec_list = [vec]
    dist_list = [self.vec_len(vec)]
    


    #add the last point as well
    for i in range(self._start_point_index, 
      min(self._start_point_index + num_of_points, len(self._path))):
      #cur_ind = min(self._start_point_index + i, len(self._path)-1)
      vec = (y[0]- self._path[i][0], y[1]- self._path[i][1])
      vec_list.append(vec)
      dist_list.append(self.vec_len(vec))
      if(dist_list[-1] + dist_list[-2] < min_dist):
        min_dist = dist_list[-1] + dist_list[-2]
        min_dist_index = i - 1
        temp_ind = i - self._start_point_index
     
    vec1 = vec_list[temp_ind]
    vec2 = vec_list[temp_ind + 1]
    p1 = self._path[min_dist_index]
    p2 = self._path[min_dist_index + 1]
    line = self.vec_deduct(p2,p1)
    
    v1_vt, v1_hr, v1_dot_vec, v1_dot = self.dist2line(line, vec1)
    
       
    return v1_vt, min_dist_index, score
  
  def gaussian(self, x, m = 0, s = 1):
    return 1/(math.sqrt(2*pi)*s)*math.e**(-0.5*(float(x-m)/s)**2)
  
  def vec_dot(self, vec1, vec2):
    return sum(p*q for p,q in zip(vec1, vec2))
  
  #need testing
  def vec_deduct(self, vec1, vec2):
    return tuple([p - q for p,q in zip(vec1, vec2)])
  
  def vec_len(self, vec):
    return math.sqrt(sum(p*p for p in vec))
  
  #return: dist to line, dist from start of line to point, 
  #        directional dot between line and vector, dot between line and vector
  def dist2line(self, line, vec1):
    dot = self.vec_dot(line, vec1)
    line_len = self.vec_len(line)
    dot_vec = tuple([dot / line_len / line_len * e for e in line])
    vertical_vec = self.vec_deduct(vec1, dot_vec)
    dist = self.vec_len(vertical_vec)
    horiz_dist_v1 = self.vec_len(dot_vec)
    return dist, horiz_dist_v1, dot_vec, dot


  def reset(self):
    self.reset_path()
    self.reset_state()
    self.reset_param()
    return np.array(self._state, dtype=np.float32)
  
  def reset_path(self):
    if self.use_random_path:
      if not self.use_hard_path:
        self._path = self.generate_easy_path(path_length)
      else:
        self._path = self.generate_random_path(path_length)
  
  def reset_state(self):
    self._x = 0
    self._y = 0
    self._t = 0
    random.seed(datetime.now())
    if(use_random_state):
        state_0 = [random.uniform(-pi/4, pi/4) for i in range(self.n - 1)] + self.vec_rotate(0,0) + [0]*(self.n - 1)
    else:
      #state_0 = [pi/6, -pi/6]+self.point_transformation([0,0])+[0, 0]
      state_0 = [0.2, -0.2] + [0]*(self.n-3) + self.vec_rotate(0,0)+ [0, 0] + [0]*(self.n-3)
    temp_path = self._path[:num_of_points]
    
    for i in range(num_of_points):
      state_0 += self.point_transformation(temp_path[i])
    self._state = state_0 + [1]
  
  def reset_param(self):
    self._episode_ended = False
    self._total_step = 0
    self._start_point_index = 1
    #this is for recording the general motion
    self.has_past = False
  
  def get_distance_reward(self, y):
    return math.sqrt((y[0] - self._x) * (y[0] - self._x) + (y[1] - self._y) * (y[1] - self._y))

  def step(self, action):
    
    self._total_step+=1
    
    if self._episode_ended:
      return self.reset()
    
    #use the snake library to simulate the movement given the command
    amp, freq = self.action2motor(action)
    initial_config = self.get_initial_config()
    y,t = self._snake_model.generate_trajectories(initial_config, 
        amplitude = amp, frequency = freq, t_val = [0, time_interval, 10])
    
    if(self.save_trajectory):
      if(not self.has_past):
        self.past_traj = y
        self.past_t = t
        self.time_counter = 1
        self.has_past = True
      else:
        self.past_traj = np.concatenate((self.past_traj, y), axis = 0)
        self.past_t = np.concatenate((self.past_t, t + time_interval*self.time_counter))
        self.time_counter +=1

    # assume we only need the last row, can also optimize the 
    # variance here if we want to control the movement in between 
    y = y[-1,:].tolist()           
    vertical_distance, min_dist_index, score = self.cal_distances_to_points(y)
    
    dist_reward = self.gaussian(vertical_distance/3)
    reward = dist_reward * score * 10.0
    if(score < 0):
      reward = 0
    
    #print(f"distance_to_line: {vertical_distance}, distance_reward: {dist_reward}, score: {score}, min_dist_index: {min_dist_index+1}")
  
    self._start_point_index = min_dist_index + 1
    
    self.update_state(y)
    
    if(self._total_step == end_step_num):
      self._episode_ended = True
      end_early = False
      
    #end the episode if it is close enough to the target 
    if ( self._x - self._path[-1][0])**2 + ( self._y - self._path[-1][1])**2 < dist_threshold :
      self._episode_ended = True
      end_early = True
    
    if self._episode_ended:
      if end_early:
        return np.array(self._state, dtype=np.float32), reward + 30, self._episode_ended, {}
      else:
        return np.array(self._state, dtype=np.float32), reward, self._episode_ended, {}
      
    else:
      return np.array(self._state, dtype=np.float32), reward, self._episode_ended, {}
    
  #generate a path of n points, each with a distance of ~1, (also add momentum so that the path won't collapse together)
  #here we are assuming that the robot always start from 0,0, which may not always be the case
  #this will/might be handled later
  def generate_random_path(self, n, point_dist = 2):
    random.seed(datetime.now())
    path = [(0,0)]
    angle = random.uniform(0,2*pi)
    x = point_dist*cos(angle)+random.uniform(-0.1,0.1)
    y = point_dist*sin(angle)+random.uniform(-0.1,0.1)
    path.append((x,y))
    prev_theta = angle
    for i in range(n-1):
      angle = prev_theta + random.uniform(-pi/8.0,pi/8.0)
      x = path[-1][0]+point_dist*cos(angle)+random.uniform(-0.1,0.1)
      y = path[-1][1]+point_dist*sin(angle)+random.uniform(-0.1,0.1)
      path.append((x,y))
      prev_theta = angle
    return path
  
  #should be easier for the robot to follow, espycially when the robot don't use random state
  def generate_easy_path(self, n, point_dist = 2):
    random.seed(datetime.now())
    path = [(0,0)]
    angle = random.uniform(pi,3*pi/4.0)
    x = cos(angle)*point_dist
    y = sin(angle)*point_dist
    path.append((x,y))
    prev_theta = angle
    for i in range(n-1):
      angle = prev_theta + random.uniform(-pi/8.0,pi/8.0)
      x = path[-1][0]+point_dist*cos(angle)
      y = path[-1][1]+point_dist*sin(angle)
      path.append((x,y))
      prev_theta = angle
    return path
  
  def point_transformation(self, point):
    x = point[0] - self._x
    y = point[1] - self._y
    x_rot = x * np.cos(-self._t) - y * np.sin(-self._t)
    y_rot = y * np.cos(-self._t) + x * np.sin(-self._t)
    return [x_rot,y_rot]

  def vec_rotate(self, x, y):
    x_rot = x * np.cos(-self._t) - y * np.sin(-self._t)
    y_rot = y * np.cos(-self._t) + x * np.sin(-self._t)
    return [x_rot, y_rot]
  
  def vec_back(self, x, y):
    x_rot = x * np.cos(self._t) - y * np.sin(self._t)
    y_rot = y * np.cos(self._t) + x * np.sin(self._t)
    return x_rot, y_rot

  def render(self):
    for i in range(len(self._path)):
      plt.plot(self._path[i][0], self._path[i][1],  'ro')
    plt.plot(self._x,self._y, 'bo')
    plt.plot(self._x+0.1*cos(self._t),self._y+0.1*sin(self._t),'bo')
    plt.show()
  
  def draw_trajectory(self, complicate = True):
    self._snake_model.plot_image(self.past_traj, self.past_t, show_pos = complicate, show_vel = complicate)
  
  def write_csv(self, csv_dir):
    self.past_t = np.asarray(self.past_t).reshape((-1,1))
    self.past_traj = np.asarray(self.past_traj)
    print(self.past_traj.shape, self.past_t.shape)
    info = np.concatenate((self.past_traj, self.past_t), axis = 1)
    np.savetxt(os.path.join(csv_dir,"test.csv"),info)
  
  #calculate new pos, calculate distance, reset points
  def update_state(self,y):
    self._x = y[0]
    self._y = y[1]
    self._t = y[2]
    
    #generate the new state
    temp_state = []

    #q1, q2
    for i in range(self.n - 1):
        temp_state.append(y[i+3])
    
    #the transformed dx, dy
    temp_state += self.vec_rotate(y[self.n+2], y[self.n+3])
    
    #dt & dq2
    temp_state.append(y[self.n+4])
    for i in range(self.n -2):
        temp_state.append(y[i + self.n + 6])

    
    for i in range (num_of_points):
      index = min(len(self._path)-1,self._start_point_index+i)
      temp_state += self.point_transformation(self._path[index])
    self._state = temp_state + [1-self._state[-1]]

  #switch the physical parameter of the snake model
  def switch_snake(self, mass = None, k_val = None, link_length = None):
    self._snake_model.switch_param(mass, k_val, link_length)

  def get_physical_params(self):
  	return self._snake_model.mass + self._snake_model.k_val + self._snake_model.link_length


if __name__ == "__main__":
    random_action = [0.4]
    #path = [(0,0),(-2,-2),(-4,-4),(-6,-6),(-8,-8),(-9,-9)]
    path = [(-2*i,-2*i) for i in range(10)]
    environment = SnakeLocomotionEnv(path, robot_link_length = 0.3, record_trajectory = True )
    obs = environment.reset()
    #print(time_step)
    cumulative_reward = 0

    for _ in range(16):
      #random_action[0] *= -1
      #print(random_action)
      obs, reward, done, _ = environment.step(random_action)
      #print(time_step)
      
      cumulative_reward += reward
      if(done):
        obs = environment.reset()
      #environment.render()
      #environment.write_csv()

    environment.draw_trajectory(complicate = True)
    #print(time_step)
    print('Final Reward = ', cumulative_reward)
