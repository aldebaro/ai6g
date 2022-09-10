import matplotlib.pyplot as plt
import os
import numpy as np

reward_mdp = np.load("./hist/rewards.npz")
reward_mdp = reward_mdp.f.rewards
reward_sac = np.load("./hist/rewards_sac.npz")
reward_sac = reward_sac.f.rewards
reward_td3 = np.load("./hist/rewards_td3.npz")
reward_td3 = reward_td3.f.rewards
reward_ppo = np.load("./hist/rewards_ppo.npz")
reward_ppo = reward_ppo.f.rewards
reward_opt = np.load("./hist/rewards_opt2.npz")
reward_opt = reward_opt.f.rewards
reward_rr = np.load("./hist/rewards_rr.npz")
reward_rr = reward_rr.f.rewards



# Histogram
w, h = plt.figaspect(0.6)
fig = plt.figure(figsize=(w, h))
plt.xlabel("Cumulative reward", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid()
plt.hist(np.sum(reward_opt, axis=1), label="IP", alpha=0.5)
plt.hist(np.sum(reward_mdp, axis=1), label="Bellman", alpha=0.5)
plt.hist(np.sum(reward_sac, axis=1), label="SAC", alpha=0.5)
plt.hist(np.sum(reward_td3, axis=1), label="TD3", alpha=0.5)
plt.hist(np.sum(reward_ppo, axis=1), label="PPO", alpha=0.5)
# plt.hist(np.sum(reward_rr, axis=1), label="RR", alpha=0.5)
fig.tight_layout()
plt.xticks(fontsize=12)
plt.legend(fontsize=12)
os.makedirs("./results", exist_ok=True)
fig.savefig(
	"./results/histogram.pdf",
	# bbox_inches="tight",
	pad_inches=0,
	format="pdf",
	dpi=1000,
)
plt.show()
# plt.close()