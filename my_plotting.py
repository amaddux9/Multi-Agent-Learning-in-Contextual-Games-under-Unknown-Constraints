#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from my_aux_functions import *

plt.close('all')
#%% 
'''
Load stored data
'''
T = 1000 

N_types = []
N_types.append(['random','random','random']) # 0
#N_types.append(['c_random','random','random'])
#N_types.append(['cGPMW','random','random']) 
N_types.append(['GPMW','random','random']) # 1
N_types.append(['zGPMW','random','random']) # 2
N_types.append(['cAdaNormalGP','random','random']) # 3
N_types.append(['czAdaNormalGP','random','random']) # 4

N_type = []
UCB_r = []
UCB_g = []
LCB_g = []
played_actions = []
obtained_payoffs = []
cum_payoffs = []
cum_constraints = []
regrets = []
weights = []

for type in range(len(N_types)):
    with open(f"Player_{N_types[type][0]}_T={T}_.pckl", 'rb') as file:
   
        N_type.append(pickle.load(file))
        r = pickle.load(file)
        g =pickle.load(file)
        runs =pickle.load(file)
        N =pickle.load(file)
        T =pickle.load(file)
        K =pickle.load(file)
        contexts =pickle.load(file)
        rev_contexts =pickle.load(file)
        version =pickle.load(file)
        UCB_r.append(pickle.load(file))
        UCB_g.append(pickle.load(file))
        LCB_g.append(pickle.load(file))
        played_actions.append(pickle.load(file))
        obtained_payoffs.append(pickle.load(file))
        cum_payoffs.append(pickle.load(file))
        cum_constraints.append(pickle.load(file))
        regrets.append(pickle.load(file))
        weights.append(pickle.load(file))

# %%
'''
Plot mean regret and mean cumulative constraint violation for the
different player types
'''

# Regrets
# random
avg_regrets_random = np.mean([[regrets[0][run][t][0] for t in range(T)] for run in range(runs)], axis = 0)
std_regrets_random = np.std([[regrets[0][run][t][0] for t in range(T)] for run in range(runs)], axis = 0)
# GPMW
avg_regrets_GPMW = np.mean([[regrets[1][run][t][1] for t in range(T)] for run in range(runs)], axis = 0)
std_regrets_GPMW = np.std([[regrets[1][run][t][1] for t in range(T)] for run in range(runs)], axis = 0)
# zGPMW
avg_regrets_zGPMW = np.mean([[regrets[2][run][t][0] for t in range(T)] for run in range(runs)], axis = 0)
std_regrets_zGPMW = np.std([[regrets[2][run][t][0] for t in range(T)] for run in range(runs)], axis = 0)
# cAdaNormalGP
avg_regrets_cAdaNormalGP = np.mean([[regrets[3][run][t][0] for t in range(T)] for run in range(runs)], axis = 0)
std_regrets_cAdaNormalGP = np.std([[regrets[3][run][t][0] for t in range(T)] for run in range(runs)], axis = 0)
# czAdaNormalGP
avg_regrets_czAdaNormalGP = np.mean([[regrets[4][run][t][0] for t in range(T)] for run in range(runs)], axis = 0)
std_regrets_czAdaNormalGP = np.std([[regrets[4][run][t][0] for t in range(T)] for run in range(runs)], axis = 0)

# Constraints
# random
avg_cum_constraints_random = np.mean([[float(cum_constraints[0][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)
std_cum_constraints_random = np.std([[float(cum_constraints[0][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)
# GPMW
avg_cum_constraints_GPMW = np.mean([[float(cum_constraints[1][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)
std_cum_constraints_GPMW = np.std([[float(cum_constraints[1][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)
# zGPMW
avg_cum_constraints_zGPMW = np.mean([[float(cum_constraints[2][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)
std_cum_constraints_zGPMW = np.std([[float(cum_constraints[2][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)
# cAdaNormalGP
avg_cum_constraints_cAdaNormalGP = np.mean([[float(cum_constraints[3][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)
std_cum_constraints_cAdaNormalGP = np.std([[float(cum_constraints[3][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)
# czAdaNormalGP
avg_cum_constraints_czAdaNormalGP = np.mean([[float(cum_constraints[4][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)
std_cum_constraints_czAdaNormalGP = np.std([[float(cum_constraints[4][run][t][0]) for t in range(T+1)] for run in range(runs)], 0)


fig = plt.figure(figsize=(10,3))
ax1 = plt.subplot2grid((1,2), (0,0), colspan=1, rowspan=2)
ax2 = plt.subplot2grid((1,2), (0,1), colspan=1, rowspan=2)

# Regret
# random
ax1.plot(np.arange(T), avg_regrets_random, color='black', marker = '*', markevery = 50, markersize = 5, label='random')
ax1.fill_between(np.arange(T), avg_regrets_random - std_regrets_random, avg_regrets_random + std_regrets_random, alpha=0.1,color='black')
# GPMW
ax1.plot(np.arange(T), avg_regrets_GPMW, color='cyan', marker = '*', markevery = 50, markersize = 5, label='GPMW')
ax1.fill_between(np.arange(T), avg_regrets_GPMW - std_regrets_GPMW, avg_regrets_GPMW + std_regrets_GPMW, alpha=0.1,color='cyan')
# zGPMW
ax1.plot(np.arange(T), avg_regrets_zGPMW, color='blue', marker = '*', markevery = 50, markersize = 5, label='z.GPMW')
ax1.fill_between(np.arange(T), avg_regrets_zGPMW - std_regrets_zGPMW, avg_regrets_zGPMW + std_regrets_zGPMW, alpha=0.1,color='blue')
# cAdaNormalGP
ax1.plot(np.arange(T), avg_regrets_cAdaNormalGP, color='magenta', marker = '*', markevery = 50, markersize = 5, label='c.AdaNormalGP')
ax1.fill_between(np.arange(T), avg_regrets_cAdaNormalGP - std_regrets_cAdaNormalGP, avg_regrets_cAdaNormalGP + std_regrets_cAdaNormalGP, alpha=0.1,color='magenta')
# czAdaNormalGP
ax1.plot(np.arange(T), avg_regrets_czAdaNormalGP, color='red', marker = '*', markevery = 50, markersize = 5, label='c.z.AdaNormalGP')
ax1.fill_between(np.arange(T), avg_regrets_czAdaNormalGP - std_regrets_czAdaNormalGP, avg_regrets_czAdaNormalGP + std_regrets_czAdaNormalGP, alpha=0.1,color='red')

ax1.set_xlim([0,T])
ax1.set_ylim([0,1])
ax1.set_xlabel('T', fontsize=12)
ax1.set_ylabel('Regret', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)


# Cumulative constraints
# random
ax2.plot(np.arange(T+1), avg_cum_constraints_random, color='black', marker = '*', markevery = 50, markersize = 5, label='random')
ax2.fill_between(np.arange(T+1), avg_cum_constraints_random - std_cum_constraints_random, avg_cum_constraints_random + std_cum_constraints_random, alpha=0.1,color='black')
# GPMW
ax2.plot(np.arange(T+1), avg_cum_constraints_GPMW, color='cyan', marker = '*', markevery = 50, markersize = 5, label='GPMW')
ax2.fill_between(np.arange(T+1), avg_cum_constraints_GPMW - std_cum_constraints_GPMW, avg_cum_constraints_GPMW + std_cum_constraints_GPMW, alpha=0.1,color='cyan')
# zGPMW
ax2.plot(np.arange(T+1), avg_cum_constraints_zGPMW, color='blue', marker = '*', markevery = 50, markersize = 5, label='z.GPMW')
ax2.fill_between(np.arange(T+1), avg_cum_constraints_zGPMW - std_cum_constraints_zGPMW, avg_cum_constraints_zGPMW + std_cum_constraints_zGPMW, alpha=0.1,color='blue')
# cAdaNormalGP
ax2.plot(np.arange(T+1), avg_cum_constraints_cAdaNormalGP, color='magenta', marker = '*', markevery = 50, markersize = 8, label='c.AdaNormalGP')
ax2.fill_between(np.arange(T+1), avg_cum_constraints_cAdaNormalGP - std_cum_constraints_cAdaNormalGP, avg_cum_constraints_cAdaNormalGP + std_cum_constraints_cAdaNormalGP, alpha=0.1,color='magenta')
# czAdaNormalGP
ax2.plot(np.arange(T+1), avg_cum_constraints_czAdaNormalGP, color='red', marker = '*', markevery = 50, markersize = 5, label='c.z.AdaNormalGP')
ax2.fill_between(np.arange(T+1), avg_cum_constraints_czAdaNormalGP - std_cum_constraints_czAdaNormalGP, avg_cum_constraints_czAdaNormalGP + std_cum_constraints_czAdaNormalGP, alpha=0.1,color='red')

ax2.set_xlim([0,T+1])
ax2.set_xlabel('T', fontsize=12)
ax2.set_ylabel('Constraints', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)


plt.tight_layout()

#plt.savefig(f'N_player_regret_constraints.png')

