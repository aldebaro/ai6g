import numpy as np
import json

initial_episode = 900
final_episode = 1000
n_steps = 1000
actions = np.zeros((final_episode-initial_episode, n_steps))
for idx, episode in enumerate(np.arange(initial_episode, final_episode)):
	with open("opt-out/ep{}.npz.out".format(episode)) as f:
		data = json.load(f)
	ue1_alloc = data["s"]["0"]
	ue2_alloc = data["s"]["1"]
	for step in np.arange(n_steps-1):
		actions[idx,step] = 0 if ue1_alloc[step]==1 else 1
# np.savez_compressed("./actions_opt.npz", actions=actions)