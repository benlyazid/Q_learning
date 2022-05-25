import numpy as np
import gym
import random
import time

L_R = 1 # Learning rate is variable in range(0, 1) used in Belman equation it's mean how much we want to update the value.
GAMMA = 1 # We set GaMMA to 1 because we are interested to next state.
ENV_NAME = "CartPole-v1" # The enviroment tht we want to include from GYM library.
STATE_NUMBER = 5 # We make very parametere of the state between 0 and 5 to make the numbers of states limited.
EPSILONE = 0.05 # This variable means how mutch we want to explore new move or exploit our move that we are already explore.
TRINING_LEN = 5000 # How mutch we want to trine our agent.
SCORE_TO_ACHEIVE = 195 # The score that We want to achieve
PINALTY = -300 # The pinalty That we will give to our agent after losing 
Q_table = {} # Where we will store are our state and action


def updtae_q_table_value(observation, env, action, state, reward):
	# Function to update the Q_table using belman equation
	#		BELMAN EQUATION
	# Q_table[state][action_chosed] = Q_table[state][action_chosed] + learning_rate * (reward_that_we_get_from_that_move +
	# 								  GAMMA * expected_reward_that_we_can_get_from_next_move - Q_table[state][action_chosed])
	state = str(state)
	state_2 = convert_observation_data(observation, env)
	if state not in Q_table:
		Q_table[state] = [0, 0]
	old_value = Q_table[state][action]
	Q_table[state][action] = old_value + L_R * (reward + GAMMA * chose_action_from_table(state_2) - old_value)
	
def	chose_action_from_table(state):
	# chose action that havae most expected reward based on the table and we have just to move left or right (0 | 1)
	state = str(state)
	if state not in Q_table:
		Q_table[state] = [0, 0]
	actions = Q_table[state]
	if actions[0] >= actions[1]:
		return 0
	return 1

def convert_observation_data(observation, env):
	# convert status to int numbers in spescefique range to make number of state limited in 5^4 = 625
	range_len = STATE_NUMBER
	low = [-2.4, -3, -0.418, -4]
	hight = [2.4, 3, 0.418, 4]
	data = []
	for i in range(len(observation)):
		a = (observation[i] - low[i]) / (hight[i] - low[i])
		a = int(a * range_len)
		data.append(a)
	return (data)

def training(env):
	global EPSILONE, L_R
	score = 0
	test_score = 0
	while test_score  < 200:
		for i in range(TRINING_LEN):
			total_reward = 0
			observation = env.reset()
			done = False
			while not done:
				state = convert_observation_data(observation, env)
				# we select random move if random number < EPSILON and just in the first 75% game (exploring new moves).
				if np.random.uniform(0, 1) < EPSILONE  and  i < TRINING_LEN - (TRINING_LEN / 4):
					action = np.random.choice(env.action_space.n)
				# Select action from table (EXPLOITING).
				else: 
					action = chose_action_from_table(state)
				observation, reward, done, info = env.step(int(action))
				# for every move our agent get 1 like a reward event in the last move, so we change the last move reward to PINALTI VALUE like.
				if done and total_reward < SCORE_TO_ACHEIVE:
					reward = PINALTY
				updtae_q_table_value(observation, env, int(action), state, reward)
				total_reward += reward
				if done:
					break
			if total_reward >= SCORE_TO_ACHEIVE:
				score += 1
			# We decrease the Learning_rate and Epsilon value after each episode.
			EPSILONE -= (EPSILONE / (TRINING_LEN))
			L_R -= (L_R / TRINING_LEN)
			if i and i % 1000 == 0:
				print("Number of game where we achive the goal in 1000 game is {}".format(score))
				test_score = score
				score = 0
			if test_score  >= 200:
				break;
	print("\nFnish trining")


# Calcule The average of 100 match, it's should pass 195.
def calcule_average(env):
	note = 0
	average = 0
	for i in range(100):
		observation = env.reset()
		score = 0
		done = False
		while not done:
			state = convert_observation_data(observation, env)
			action = chose_action_from_table(state) #swap #  env.action_space.sample() # your agent here (this takes random actions)
			observation, reward, done, info = env.step(action)
			convert_observation_data(observation, env)
			score += reward
			if done:
				break
		if score >= 195:
			note += 1
		average += score
	average /= 100
	print("\nAVERAGE IS {} and  note {}".format(average, note))

def simulate_our_agent(env):
	print("\nSTART SIMULATION\n")
	observation = env.reset()
	score = 0
	done = False
	while not done:
		env.render()
		time.sleep(0.05)
		state = convert_observation_data(observation, env)
		action = chose_action_from_table(state)
		observation, reward, done, info = env.step(action)
		convert_observation_data(observation, env)
		score += reward
		if done:
			break
	print("\n OUR SCORE IS {}".format(score))

# SET ENVIREMENT
env = gym.make(ENV_NAME)
total = 0
done = False
env.seed(0)
np.random.seed(0)

training(env)
calcule_average(env)
simulate_our_agent(env)