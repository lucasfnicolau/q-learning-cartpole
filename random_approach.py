import gym
import random
import matplotlib
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')

def random_games(iterations=100):
	episodes = []

	# Each of this episode is its own game.
	for episode in range(iterations):
		episode_length = 0

		env.reset()

		print('\nEpisode', episode)
		# This is each frame, up to 500... but we won't
		# make it that far with random.
		for t in range(500):

			# This will display the environment.
			# Only display if you really want to see it.
			# Takes much longer to display it.
			# env.render()

			# This will just create a simple action in any environment.
			# In this environment, the action can be 0 or 1, which is left or right.
			action = env.action_space.sample()

			# This executes the environment with an action,
			# and returns the observation of the environment,
			# the reward, if the env is over, and other info.
			next_state, reward, done, info = env.step(action)

			episode_length += 1

			# Let's print everything in one line:
			print(t, next_state, reward, done, action)

			if done:
				episodes.append(episode_length)
				break
	return episodes

def plot(episodes):
	plt.title('Abordagem aleatória')
	plt.xlabel('Episódios')
	plt.ylabel('Número de movimentos')
	plt.ylim(top=500)
	plt.plot(episodes)
	plt.show()

episodes = random_games()
plot(episodes)
env.close()
