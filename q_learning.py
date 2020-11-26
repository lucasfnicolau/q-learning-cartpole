# TCC - Ciência da Computação / Mackenzie - 2020
#
# Leonardo Shun Mendes de Rosa (LeShuno)
# Lucas Fernandez Nicolau (lucasfnicolau)
# Thiago Akio Orlando Kumagai (KumagaiT)
#
# Referências de código:
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
# https://github.com/vmayoral/basic_reinforcement_learning/blob/master/tutorial4/q-learning-gym-1.py
# https://medium.com/@flomay/using-q-learning-to-solve-the-cartpole-balancing-problem-c0a7f47d3f9d

import gym
from gym import wrappers
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import clear_output

GAMMA = 1
epsilon = 1
alpha = 1

env = gym.make('CartPole-v1')

q_table = {}
bins = [
	np.zeros(3),
	np.zeros(3),
	np.zeros(6),
	np.zeros(12)
]

successful_recording = False

def get_action(dict):
	return q_max(dict)[0]

def get_max_q(dict):
	return q_max(dict)[1]

def q_max(dict):
	max_val = float('-inf')
	for key, val in dict.items():
		if val > max_val:
			max_val = val
			max_key = key
	return max_key, max_val

def create_bins():
	bins[0] = np.linspace(-4.8, 4.8, bins[0].size) # posição do carro
	bins[1] = np.linspace(-5.0, 5.0, bins[1].size) # velocidade do carro
	bins[2] = np.linspace(-.418, .418, bins[2].size) # ângulo do pêndulo
	bins[3] = np.linspace(-5.0, 5.0, bins[3].size) # velocidade do pêndulo

def assign_bins(observation, bins):
	state = np.zeros(4)
	for i in range(4):
		state[i] = np.digitize(observation[i], bins[i])
	return state

def get_state(state):
	string_state = ''.join(str(int(s)) for s in state)
	return string_state

def get_all_states():
	states = []
	for i in range(bins[0].size):
		for j in range(bins[1].size):
			for k in range(bins[2].size):
				for l in range(bins[3].size):
					states.append(str(i) + str(j) + str(k) + str(l))
	return states

def initialize_Q():
	all_states = get_all_states()
	for state in all_states:
		q_table[state] = {}
		for action in range(env.action_space.n):
			q_table[state][action] = 0

def train(GAMMA, alpha, epsilon, iterations=3001):
	print('\nComeçando treino...')

	successful_episodes = 0

	for i in range(iterations):
		observation = env.reset()
		state = get_state(assign_bins(observation, bins))

		done = False

		episode_length = 0

		while not done:
			if random.uniform(0, 1) < epsilon:
				action = env.action_space.sample() # 'Explore'
			else:
				action = get_action(q_table[state]) # 'Exploit'

			episode_length += 1

			observation, reward, done, info = env.step(action)
			next_state = get_state(assign_bins(observation, bins))

			if done:
				if episode_length < 500:
					reward = -300
				else:
					successful_episodes += 1

			old_value = q_table[state][action]
			next_max = get_max_q(q_table[next_state])

			new_value = alpha * (reward + GAMMA * next_max - old_value)
			q_table[state][action] += new_value

			state = next_state

			epsilon = 1.0 / np.sqrt(i + 1)
			alpha = 1.0 / np.sqrt(i + 1)

		if i % 1000 == 0 and i > 0:
			clear_output(wait=True)
			print(f'Episódio: {i} | Número de sucessos: {successful_episodes}/1000')
			successful_episodes = 0

	print('Treino finalizado.\n')

def evaluate(env, successful_recording, iterations=100):
	total_episodes_length = 0
	episodes = []

	for i in range(iterations):
		if not successful_recording:
			env = wrappers.Monitor(env, './video', force=True)
			env.render()

		observation = env.reset()
		state = get_state(assign_bins(observation, bins))

		episode_length, reward = 0, 0

		done = False

		while not done:
			action = get_action(q_table[state])

			observation, reward, done, info = env.step(action)
			state = get_state(assign_bins(observation, bins))

			episode_length += 1

			if done:
				episodes.append(episode_length)
				env.close()
				if episode_length == 500:
					successful_recording = True

		total_episodes_length += episode_length

	print(f'\nResultados após {iterations} episódios:')
	print(f'Média de duração por episódio: {total_episodes_length / iterations}\n')
	return episodes

def plot(episodes):
	plt.title('Q-Learning')
	plt.xlabel('Episódios')
	plt.ylabel('Número de movimentos')
	plt.ylim(top=500)
	plt.plot(episodes, linewidth=10.0)
	plt.show()

if __name__ == "__main__":
	create_bins()
	initialize_Q()
	train(GAMMA, alpha, epsilon)
	episodes = evaluate(env, successful_recording)
	plot(episodes)
