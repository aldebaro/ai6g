import numpy as np

ue1_mob_file = np.load("./src/mobility_ue0.npz")
ue1_pos_actions_prob = ue1_mob_file.f.pos_actions_prob
ue1_matrix_pos_prob = ue1_mob_file.f.matrix_pos_prob

ue2_mob_file = np.load("./src/mobility_ue1.npz")
ue2_pos_actions_prob = ue2_mob_file.f.pos_actions_prob
ue2_matrix_pos_prob = ue2_mob_file.f.matrix_pos_prob

grid_size = 6
actions = 5
joint_prob = np.zeros((grid_size,grid_size,grid_size,grid_size))
actions_move = np.array([
            [-1, 0],
            [0, 1],
            [1, 0],
            [0, -1],
            [0, 0],
        ])  # Up, right, down, left, stay
basestation_pos = np.array([5,0])

for ue1_row in np.arange(grid_size):
	for ue1_col in np.arange(grid_size):
		for ue2_row in np.arange(grid_size):
			for ue2_col in np.arange(grid_size):
				if ue1_row==ue2_row and ue1_col==ue2_col:
					continue
				for ue1_action in np.arange(actions):
					for ue2_action in np.arange(actions):
						new_ue1_pos = np.array([ue1_row, ue1_col])+actions_move[ue1_action]
						new_ue2_pos = np.array([ue2_row, ue2_col])+actions_move[ue2_action]
						if np.sum((new_ue1_pos>grid_size))>1 or \
									np.sum(new_ue1_pos<0)>1 or \
									np.sum((new_ue1_pos>grid_size))>1 or \
									np.sum(new_ue1_pos<0)>1 or \
									np.array_equal(new_ue1_pos,new_ue2_pos) is True or \
									np.array_equal(new_ue1_pos, basestation_pos) is True or \
									np.array_equal(new_ue2_pos, basestation_pos) is True:
							joint_prob[ue1_row, ue1_col, ue2_row, ue2_col] = 0
						ue1_matrix_pos_prob[ue1_row,ue1_col]