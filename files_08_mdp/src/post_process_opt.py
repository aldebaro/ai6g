import numpy as np

n_steps = 1000
initial_episode = 900
n_episodes = 100
rewards = np.zeros((100, 1000))

for idx, episode in enumerate(np.arange(initial_episode, initial_episode+n_episodes)):
	file = np.load("./opt-out/opt-out-ep{}.npz".format(episode))
	pkt_loss_ue1 = file.f.arr_0
	pkt_loss_ue2 = file.f.arr_1
	rewards[idx] -= np.append(0, pkt_loss_ue1+pkt_loss_ue2)
np.savez_compressed("./hist/rewards_opt.npz", rewards=rewards)