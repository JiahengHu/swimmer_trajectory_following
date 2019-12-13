import gym
# gym.envs.register(
#      id='Snake-v0',
#      entry_point='snake_env.gym_snake_env:SnakeLocomotionEnv',
#      max_episode_steps=50,
#      #kwargs={'size' : 1, 'init_state' : 10., 'state_bound' : np.inf},
# )
# env = gym.make('Snake-v0')

env = gym.make('Hopper-v2')
env.reset()
while True:
	env.step(1)
	#env.render()