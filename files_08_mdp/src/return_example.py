import numpy as np
r=[-1, 2, 6, 3, 2]
gamma=0.5 # discount rate
# Calculate G for time t=0
n=[0, 1, 2, 3, 4]
print('G0=', np.sum(r * np.power(gamma,n)))
print('G1=', np.sum(r[1:] * np.power(gamma,n[:-1])))
# Calculate G recursively and backwards for all time t
Gt=0 # initialize G_T = 0
for t in np.arange(4,-1,-1):
	Gtp1=r[t]+gamma*Gt # recursive equation
	print('G' + str(t) + '=' + str(Gtp1))
	Gt=Gtp1 # update for next iteration
	
