import numpy as np

from FiniteMDP import FiniteMDP
from UserSchedulingEnv import UserSchedulingEnv

# print('1')
episodes = np.arange(900, 1000)
n_steps = 1000
se = np.load("./src/spec_eff_matrix.npz")
se = se.f.spec_eff_matrix
# env = UserSchedulingEnv()
file_states_actions = np.load("./src/states_actions.npz", allow_pickle=True)
indexGivenStateDictionary = file_states_actions.f.indexGivenStateDictionary.item()
file_optimal_values = np.load("./src/optimal_values.npz")
optimal_policy = file_optimal_values.f.optimal_policy
# mdp = FiniteMDP(env)
# print('1.5')
rewards = np.zeros((len(episodes), n_steps))
n_users = 2

actions = np.load("./actions_opt.npz")
actions = actions.f.actions
buffer_size = 3
num_incoming_packets_per_time_slot = 2
rewards = np.zeros((len(episodes), n_steps))
for n_episode, episode in enumerate(episodes):
	file_pos = np.load("./mobility_traces/ep{}.npz".format(episode))
	for step in np.arange(n_steps):
		print("Episode {}, step {}".format(n_episode, step))
		pos_ues = ((file_pos.f.ue1[step][0], file_pos.f.ue1[step][1]), (file_pos.f.ue2[step][0], file_pos.f.ue2[step][1]))
		if step == 0:
			buffers = np.array([0, 0])
		state = (pos_ues, tuple(buffers))
		# prob_actions = optimal_policy[indexGivenStateDictionary[state]]
		# chosen_user = np.random.choice(2, p=prob_actions) #in this case, the action is the user
		chosen_user = int(actions[n_episode, step])
		number_dropped_packets = 0
		for user in np.arange(n_users):
			if user == chosen_user:
				#get the channels spectral efficiency (SE)
				chosen_user_position = pos_ues[user]

				se_chosen_ue = se[int(chosen_user_position[0]),int(chosen_user_position[1])]
				#based on selected (chosen) user, update its buffer
				transmitRate = se_chosen_ue #transmitted packets 
				buffers[chosen_user] -= transmitRate #decrement buffer of chosen user
				buffers[buffers<0] = 0
			buffers[user] += num_incoming_packets_per_time_slot #arrival of new packets

			#check if overflow
			#in case positive, limit the buffers to maximum capacity
			number_dropped_packets = buffers[user] - buffer_size
			number_dropped_packets = 0 if number_dropped_packets < 0 else number_dropped_packets

			#saturate buffer levels
			buffers[user] = buffer_size if buffers[user]>buffer_size else buffers[user]

			# calculate rewards
			rewards[n_episode, step] -= number_dropped_packets
# np.savez_compressed("./hist/rewards_opt2.npz".format(episode), rewards=rewards)
