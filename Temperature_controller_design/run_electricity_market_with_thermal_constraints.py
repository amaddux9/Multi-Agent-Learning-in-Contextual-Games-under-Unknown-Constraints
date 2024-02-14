#%%
import numpy as np
import itertools
import pickle
import GPy
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from funcs_thermal_control_cost import *
from funcs_player_type_dynamics import *
from funcs_plotting import *


#%%
# Store game as a class (N players use no-regret algorithm and learn constraints)
class GameData:
     def __init__(self,T,N,Household, true_temp_deviation, true_energy_hourly, true_cost):
        self.Household = Household
        self.true_temp_deviation = true_temp_deviation
        self.true_energy_hourly = true_energy_hourly
        self.true_cost = true_cost
        self.Played_actions = []
        self.Obtained_losses = [] 
        self.Cum_losses =  []
        self.Consumed_energy = []
        self.Obtained_constraints = []
        self.Cum_constraints = [[0]*N]
        self.Regrets =  []
        self.history_weights = []
        self.history_lcb_g = []
        self.time = T
        self.N = N 
        

def RunGame(T,N,types,Household, params_args, sim_days, dev_threshold, sigma0, sigma1, true_temp_deviation, true_energy_hourly, true_cost):
    
    P, SP_day, shift_night = params_args
    len_P = len(P)
    len_SP_day = len(SP_day)
    len_shift_night = len(shift_night)

    noises_l = np.random.normal(0,sigma0,T)
    noises_g = np.random.normal(0,sigma1,T)

    Game_data = GameData(T,N,Household,true_temp_deviation, true_energy_hourly, true_cost)
    Game_data.LCB_vectors_l = []
    Game_data.LCB_vectors_g = []

    
    # Define kernel for loss function
    
    max_dev = True
    if max_dev:
        # Define kernel for cost function
        var_loss = [10.9, 9.56, 5]
        bias_loss = [40, 31, 36]
        k_loss = []
        for i in range(N):
            k_loss.append(GPy.kern.Poly(input_dim= sim_days*24*N, variance=var_loss[i], bias=bias_loss[i], order=1))
        # Define kernel for constraint function
        var_g = [3.0, 3.14, 3.06] 
        l_g = [[1, 1, 1],
               [1,1,1],
               [1,1,1]]
        
        k_g = []
        for i in range(N):
            k_g.append(GPy.kern.RBF(input_dim=3, variance=var_g[i], lengthscale=l_g[i], active_dims=range(3), ARD= True)) 
    else:
        # Define kernel for cost function
        var_loss = [12, 32, 10]
        bias_loss = [40, 31, 36]
        k_loss = []
        for i in range(N):
            k_loss.append(GPy.kern.Poly(input_dim= sim_days*24*N, variance=var_loss[i], bias=bias_loss[i], order=1))
        # Define kernel for constraint function
        var_g = [2.0, 1.848, 0.087] 
        l_g = [[2.0406704 , 7.52084585, 4.86786981],
            [1.89473781, 7.57571923, 4.46603736],
            [8.36248439, 3.17080272, 2.92096496]]
        k_g = []
        for i in range(N):
            k_g.append(GPy.kern.RBF(input_dim=3, variance=var_g[i], lengthscale=l_g[i], active_dims=range(3), ARD= True)) 
        
    Player_type = list(range(N))  #list of all players 
    min_loss = []
    loss_range = []
    for i in range(N):
        # these values are derived from obj_func
        min_loss.append( np.array(min(true_cost[i])))
        loss_range.append( np.array(max(true_cost[i]) - min(true_cost[i])) ) 
        Game_data.Cum_losses.append( np.zeros(len_P * len_SP_day * len_shift_night) )
        
        if types[i] == "cAdaNormalGP": 
            Player_type[i] = Player_cAdaNormalGP(params_args,i,min_loss[i], loss_range[i], k_loss[i], k_g[i], dev_threshold)
        if types[i] == 'GPMW':
            Player_type[i] = Player_GPMW(T, params_args,i, min_loss[i], loss_range[i], k_loss[i])
                                        
   
    for t in range(T):

        if (t % 50 == 0):
            print('t= ' + str(t))
        
        # Feasibility check and played action
        Game_data.Played_actions.append(  [None]*N )
        Game_data.history_weights.append([0]*N)
        Game_data.history_lcb_g.append([0]*N)

        for i in range(N):

            # Feasibility Check
            if (types[i] == 'cAdaNormalGP'):
                if len([ai for ai in Player_type[i].lcb_constraint_est if (ai<= dev_threshold)]) == 0:
                    sys.exit("No feasible actions available.")  

            
            # Played action
            Game_data.history_weights[t][i] = Player_type[i].weights
            if (types[i] == 'cAdaNormalGP'):
                Game_data.history_lcb_g[t][i] = Player_type[i].lcb_constraint_est
            Game_data.Played_actions[t][i] =  Player_type[i].sample_action() 


        # Assign payoffs and compute regrets
        Game_data.Obtained_losses.append([None]*N) 
        Game_data.Consumed_energy.append([None]*N)
        Game_data.Obtained_constraints.append([None]*N)
        Game_data.Regrets.append( [None]*N )          
        Game_data.Cum_constraints.append([0]*N)

        action = Game_data.Played_actions[t]
        P_0 = P.index(action[0][0])
        SP_day_0 = SP_day.index(action[0][1])
        shift_night_0 = shift_night.index(action[0][2])
        P_1 = P.index(action[1][0])
        SP_day_1 = SP_day.index(action[1][1])
        shift_night_1 = shift_night.index(action[1][2])
        P_2 = P.index(action[2][0])
        SP_day_2 = SP_day.index(action[2][1])
        shift_night_2 = shift_night.index(action[2][2])    
        for i in range(N):    
            if i == 0:
                Game_data.Consumed_energy[t][i] = Game_data.true_energy_hourly[i][len_SP_day*len_shift_night*P_0 + len_shift_night* SP_day_0 + shift_night_0]
                Game_data.Obtained_constraints[t][i] = Game_data.true_temp_deviation[i][len_SP_day*len_shift_night*P_0 + len_shift_night* SP_day_0 + shift_night_0]
            elif i == 1:
                Game_data.Consumed_energy[t][i] = Game_data.true_energy_hourly[i][len_SP_day*len_shift_night*P_1 + len_shift_night* SP_day_1 + shift_night_1]
                Game_data.Obtained_constraints[t][i] = Game_data.true_temp_deviation[i][len_SP_day*len_shift_night*P_1 + len_shift_night* SP_day_1 + shift_night_1]
            else:
                Game_data.Consumed_energy[t][i] = Game_data.true_energy_hourly[i][len_SP_day*len_shift_night*P_2 + len_shift_night* SP_day_2 + shift_night_2]
                Game_data.Obtained_constraints[t][i] = Game_data.true_temp_deviation[i][len_SP_day*len_shift_night*P_2 + len_shift_night* SP_day_2 + shift_night_2]
            
            Game_data.Cum_constraints[t+1][i] = np.array(Game_data.Cum_constraints[t][i] + max(0,Game_data.Obtained_constraints[t][i]))
            
            Game_data.Obtained_losses[t][i] = Game_data.true_cost[i][
                len_P**2*len_SP_day**3*len_shift_night**3*P_0 + len_P**2*len_SP_day**2*len_shift_night**3*SP_day_0 + len_P**2*len_SP_day**2*len_shift_night**2*shift_night_0 +
                                                                     len_P*len_SP_day**2*len_shift_night**2*P_1 + len_P*len_SP_day*len_shift_night**2*SP_day_1 + len_P*len_SP_day*len_shift_night*shift_night_1+
                                                                     len_SP_day*len_shift_night*P_2 + len_shift_night*SP_day_2 + shift_night_2
                                                                     ]
            
            
            idx_list = np.array(list(itertools.product(range(len_P),range(len_SP_day),range(len_shift_night))))
            idx_count = 0
            for idx in idx_list:
                if (types[i] == 'cAdaNormalGP'):
                    if (Game_data.true_temp_deviation[i][idx_count] <= dev_threshold):
                        if i == 0:
                            loss_unscaled = Game_data.true_cost[i][len_P**2*len_SP_day**3*len_shift_night**3*idx[0] + len_P**2*len_SP_day**2*len_shift_night**3*idx[1] + len_P**2*len_SP_day**2*len_shift_night**2*idx[2] +
                                                                     len_P*len_SP_day**2*len_shift_night**2*P_1 + len_P*len_SP_day*len_shift_night**2*SP_day_1 + len_P*len_SP_day*len_shift_night*shift_night_1+
                                                                     len_SP_day*len_shift_night*P_2 + len_shift_night*SP_day_2 + shift_night_2]
                            Game_data.Cum_losses[i][idx_count] = np.array(
                                                            Game_data.Cum_losses[i][idx_count] + loss_unscaled
                                                                )
                        if i == 1:
                            loss_unscaled = Game_data.true_cost[i][len_P**2*len_SP_day**3*len_shift_night**3*P_0 + len_P**2*len_SP_day**2*len_shift_night**3*SP_day_0 + len_P**2*len_SP_day**2*len_shift_night**2*shift_night_0 +
                                                                     len_P*len_SP_day**2*len_shift_night**2*idx[0] + len_P*len_SP_day*len_shift_night**2*idx[1] + len_P*len_SP_day*len_shift_night*idx[2]+
                                                                     len_SP_day*len_shift_night*P_2 + len_shift_night*SP_day_2 + shift_night_2]
                            Game_data.Cum_losses[i][idx_count] = np.array(
                                                            Game_data.Cum_losses[i][idx_count] + loss_unscaled
                                                                )   
                        if i == 2:
                            loss_unscaled = Game_data.true_cost[i][len_P**2*len_SP_day**3*len_shift_night**3*P_0 + len_P**2*len_SP_day**2*len_shift_night**3*SP_day_0 + len_P**2*len_SP_day**2*len_shift_night**2*shift_night_0 +
                                                                     len_P*len_SP_day**2*len_shift_night**2*P_1 + len_P*len_SP_day*len_shift_night**2*SP_day_1 + len_P*len_SP_day*len_shift_night*shift_night_1+
                                                                     len_SP_day*len_shift_night*idx[0] + len_shift_night*idx[1] + idx[2]]
                            Game_data.Cum_losses[i][idx_count] = np.array(
                                                            Game_data.Cum_losses[i][idx_count] + loss_unscaled
                                                                )
                elif (types[i] == 'GPMW'):
                    if i == 0:
                        loss_unscaled = Game_data.true_cost[i][len_P**2*len_SP_day**3*len_shift_night**3*idx[0] + len_P**2*len_SP_day**2*len_shift_night**3*idx[1] + len_P**2*len_SP_day**2*len_shift_night**2*idx[2] +
                                                                    len_P*len_SP_day**2*len_shift_night**2*P_1 + len_P*len_SP_day*len_shift_night**2*SP_day_1 + len_P*len_SP_day*len_shift_night*shift_night_1+
                                                                    len_SP_day*len_shift_night*P_2 + len_shift_night*SP_day_2 + shift_night_2]
                        Game_data.Cum_losses[i][idx_count] = np.array(
                                                        Game_data.Cum_losses[i][idx_count] + loss_unscaled
                                                            )
                    if i == 1:
                        loss_unscaled = Game_data.true_cost[i][len_P**2*len_SP_day**3*len_shift_night**3*P_0 + len_P**2*len_SP_day**2*len_shift_night**3*SP_day_0 + len_P**2*len_SP_day**2*len_shift_night**2*shift_night_0 +
                                                                    len_P*len_SP_day**2*len_shift_night**2*idx[0] + len_P*len_SP_day*len_shift_night**2*idx[1] + len_P*len_SP_day*len_shift_night*idx[2]+
                                                                    len_SP_day*len_shift_night*P_2 + len_shift_night*SP_day_2 + shift_night_2]
                        Game_data.Cum_losses[i][idx_count] = np.array(
                                                        Game_data.Cum_losses[i][idx_count] + loss_unscaled
                                                            )   
                    if i == 2:
                        loss_unscaled = Game_data.true_cost[i][len_P**2*len_SP_day**3*len_shift_night**3*P_0 + len_P**2*len_SP_day**2*len_shift_night**3*SP_day_0 + len_P**2*len_SP_day**2*len_shift_night**2*shift_night_0 +
                                                                    len_P*len_SP_day**2*len_shift_night**2*P_1 + len_P*len_SP_day*len_shift_night**2*SP_day_1 + len_P*len_SP_day*len_shift_night*shift_night_1+
                                                                    len_SP_day*len_shift_night*idx[0] + len_shift_night*idx[1] + idx[2]]
                        Game_data.Cum_losses[i][idx_count] = np.array(
                                                        Game_data.Cum_losses[i][idx_count] + loss_unscaled
                                                            )
                idx_count = idx_count + 1
            
            if (types[i] == 'cAdaNormalGP'):
                a_star = np.argmin([Game_data.Cum_losses[i][a] for a in range(len(idx_list)) if (Game_data.true_temp_deviation[i][a] <= dev_threshold)])
            elif (types[i] == 'GPMW'):
                a_star = np.argmin([Game_data.Cum_losses[i][a] for a in range(len(idx_list))])
            l_star = Game_data.Cum_losses[i][a_star]
            l_cum = sum([ Game_data.Obtained_losses[x][i] for x in range(t+1)])
            Game_data.Regrets[t][i] = (l_cum - l_star)  / (t+1)
        
            
        " Update players next mixed strategy "
        for i in range(N):
                
            if Player_type[i].type == "cAdaNormalGP":
                last_energy_consumption = Game_data.Consumed_energy[t]
                current_energy_hourly = Game_data.true_energy_hourly[i]
                last_loss = [Game_data.Obtained_losses[t][i] + noises_l[t] ]
                last_own_action = Game_data.Played_actions[t][i] 
                last_constraint = [Game_data.Obtained_constraints[t][i] + noises_g[t]]
                # Update GP posteriors
                Player_type[i].GP_update_l(last_energy_consumption, last_loss, current_energy_hourly, sim_days, sigma0)
                Player_type[i].GP_update_g(last_own_action, last_constraint, sigma1)
                # Update weights
                Player_type[i].Update()
                
            if Player_type[i].type == "GPMW":
                last_energy_consumption = Game_data.Consumed_energy[t]
                current_energy_hourly = Game_data.true_energy_hourly[i]
                last_loss = [Game_data.Obtained_losses[t][i] + noises_l[t] ]
                # Update GP posteriors
                Player_type[i].GP_update_l(last_energy_consumption, last_loss, current_energy_hourly, sim_days, sigma0)
                # Update weights
                Player_type[i].Update()
            
                
    for i in range(N):
        Game_data.LCB_vectors_l.append(Player_type[i].lcb_loss_est)
        if (types[i] == 'cAdaNormalGP'):
            Game_data.LCB_vectors_g.append(Player_type[i].lcb_constraint_est)


    return Game_data , Player_type       


#%%
#################### Run simulation #################################

# number of rounds
T = 500
# noise on observations
sigma0 = 10
sigma1 = 0.01
# weather data
weather = "CH_BS_Basel"

# number of agents
N = 3
N_types = []
#N_types.append(['cAdaNormalGP','cAdaNormalGP','cAdaNormalGP'])
N_types.append(['GPMW', 'GPMW', 'GPMW'])

# Initialize player
Player = list(range(N))
SP_night = [21, 22, 19]
shift_day = [8, 6, 7] # equivalent to nighttime_end
for i in range(N):
    Player[i] = Household(SP_night[i], shift_day[i])



# Action set of each player
'''
P - Control gain of P controller, type: list
SP_day - set point temperature during the day, type: list
shift_night - night time shift, type: list
'''

testing = False
if testing:
    P = [0.2, 0.5, 0.8, 1.2]
    SP_day = [17, 18, 19]
    shift_night = [16,18,20]
else:
    P = [0.2,0.4, 0.6,0.8,1.0, 1.2,1.4,1.6,1.8,2.0,2.5,3.0]
    SP_day = [16,17,18,19,20,21]
    shift_night = [16, 17, 18, 19, 20]

params_args = (P, SP_day, shift_night)
len_P = len(P)
len_SP_day = len(SP_day)
len_shift_night = len(shift_night)

# number of simulation days
sim_days = 2
# temperature deviation threshold
dev_threshold = 0.1

#%% 
'''
Load the true energy consumption and temperature deviation 
of each player for all paramter combinations.
Load the true cost function of each player for all paramter 
combinations.  
'''
use_true_cost = False
seed = 8
if testing:
    with open(f"game_data_constraint_test_seed={seed}.pckl", 'rb') as file:
        true_temp_deviation = pickle.load(file)
        true_energy_hourly = pickle.load(file)
        true_room_temp = pickle.load(file)
    with open(f"game_data_cost_test_seed={seed}.pckl", 'rb') as file:
        true_cost = pickle.load(file)
else:
    with open(f"game_data_max_constraint0_seed={seed}.pckl", 'rb') as file:
        true_temp_deviation_0 = pickle.load(file)
        true_energy_hourly_0 = pickle.load(file)
        true_room_temp_0 = pickle.load(file)
    with open(f"game_data_max_constraint1_seed={seed}.pckl", 'rb') as file:
        true_temp_deviation_1 = pickle.load(file)
        true_energy_hourly_1 = pickle.load(file)
        true_room_temp_1 = pickle.load(file)
    with open(f"game_data_max_constraint2_seed={seed}.pckl", 'rb') as file:
        true_temp_deviation_2 = pickle.load(file)
        true_energy_hourly_2 = pickle.load(file)
        true_room_temp_2 = pickle.load(file)
    with open(f"game_data_max_constraint3_seed={seed}.pckl", 'rb') as file:
        true_temp_deviation_3 = pickle.load(file)
        true_energy_hourly_3 = pickle.load(file)
        true_room_temp_3 = pickle.load(file)
    if use_true_cost:
        with open(f"game_data_cost0_integer.pckl", 'rb') as file:
            true_cost0 = pickle.load(file)
        with open(f"game_data_cost1_integer.pckl", 'rb') as file:
            true_cost1 = pickle.load(file)
        with open(f"game_data_cost2_integer.pckl", 'rb') as file:
            true_cost2 = pickle.load(file)
        true_cost = [true_cost0, true_cost1, true_cost2]
    else:
        with open(f'true_feasible_cost.pckl', 'rb') as file:
            true_cost = pickle.load(file)
    true_temp_deviation = []
    true_energy_hourly = []
    true_room_temp = []
    for i in range(N):
        true_temp_deviation.append(true_temp_deviation_0[i] + true_temp_deviation_1[i] + 
                                true_temp_deviation_2[i] + true_temp_deviation_3[i]) 
        true_energy_hourly.append(true_energy_hourly_0[i] + true_energy_hourly_1[i] +
                                true_energy_hourly_2[i] + true_energy_hourly_3[i])
        true_room_temp.append(true_room_temp_0[i] + true_room_temp_1[i] +
                            true_room_temp_2[i] + true_room_temp_3[i])


#%%
seed = 8
np.random.seed(seed)

# Run game
Games_data, Player_type = RunGame(T,N,N_types[0],Player, params_args, sim_days, dev_threshold,
                                   sigma0, sigma1, true_temp_deviation, true_energy_hourly, true_cost)

LCB_loss = []
LCB_g = []
played_actions = []
obtained_loss = []
cum_loss = []
cum_constraints = []
consumed_ernergy = []
regrets = []
weights = []
history_weights = []
history_LCB_g = []
LCB_loss.append(Games_data.LCB_vectors_l)
LCB_g.append(Games_data.LCB_vectors_g)
played_actions.append(Games_data.Played_actions)
obtained_loss.append(Games_data.Obtained_losses)
cum_loss.append(Games_data.Cum_losses)
consumed_ernergy.append(Games_data.Consumed_energy)
cum_constraints.append(Games_data.Cum_constraints)
regrets.append(Games_data.Regrets)
for i in range(N):
    weights.append(Player_type[i].weights)
history_weights.append(Games_data.history_weights)
history_LCB_g.append(Games_data.history_lcb_g)

with open(f'{N_types[0]}_max_T_{T}.pckl', 'wb') as file:
    pickle.dump(N_types , file)
    pickle.dump(params_args, file)
    pickle.dump(Player, file)
    pickle.dump(sim_days, file)
    pickle.dump(dev_threshold, file)
    pickle.dump(N, file)
    pickle.dump(T, file)
    pickle.dump(LCB_loss, file)
    pickle.dump(LCB_g, file)
    pickle.dump(played_actions, file)
    pickle.dump(obtained_loss, file)
    pickle.dump(cum_loss, file)
    pickle.dump(cum_constraints, file)
    pickle.dump(consumed_ernergy, file)
    pickle.dump(regrets, file)
    pickle.dump(weights, file)
    pickle.dump(history_weights, file)
    pickle.dump(history_LCB_g, file)

#%%
#################### Plotting #############################

import numpy as np
import itertools
import pickle
import GPy
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from funcs_thermal_control_cost import *
from funcs_player_type_dynamics import *
from funcs_plotting import *

#%%

# number of rounds
T = 500
# noise on observations
sigma0 = 10
sigma1 = 0.01
# weather data
weather = "CH_BS_Basel"

# number of agents
N = 3
N_types = []
N_types.append(['cAdaNormalGP','cAdaNormalGP','cAdaNormalGP'])
N_types.append(['GPMW','GPMW','GPMW'])


# Initialize player
Player = list(range(N))
SP_night = [21, 22, 19]
shift_day = [8, 6, 7] # equivalent to nighttime_end
for i in range(N):
    Player[i] = Household(SP_night[i], shift_day[i])



# Action set of each player
'''
P - Control gain of P controller, type: list
SP_day - set point temperature during the day, type: list
shift_night - night time shift, type: list
'''
testing = False
if testing:
    P = [0.2, 0.5, 0.8, 1.2]
    SP_day = [17, 18, 19]
    shift_night = [16,18,20]
else:
    P = [0.2,0.4, 0.6,0.8,1.0, 1.2,1.4,1.6,1.8,2.0,2.5,3.0]
    SP_day = [16,17,18,19,20,21]
    shift_night = [16, 17, 18, 19, 20]

params_args = (P, SP_day, shift_night)
len_P = len(P)
len_SP_day = len(SP_day)
len_shift_night = len(shift_night)

# number of simulation days
sim_days = 2
# temperature deviation threshold
dev_threshold = 0.1

# Open data of cAdaNormalGP player
with open(f'{N_types[0]}_max_T_{T}.pckl', 'rb') as file:
    N_types_old1 = pickle.load(file)
    params_args = pickle.load(file)
    Player = pickle.load(file)
    sim_days = pickle.load(file)
    dev_threshold = pickle.load(file)
    N = pickle.load( file)
    T = pickle.load(file)
    LCB_loss = pickle.load(file)
    LCB_g = pickle.load(file)
    played_actions = pickle.load(file)
    obtained_loss = pickle.load(file)
    cum_loss = pickle.load(file)
    cum_constraints = pickle.load(file)
    consumed_ernergy = pickle.load(file)
    regrets = pickle.load(file)
    weights = pickle.load(file)
    history_weights = pickle.load(file)
    history_LCB_g = pickle.load(file)

# Open data of GPMW player
with open(f'{N_types[1]}_max_T_{T}.pckl', 'rb') as file:
    N_types_old2 = pickle.load(file)
    params_args = pickle.load(file)
    Player = pickle.load(file)
    sim_days = pickle.load(file)
    dev_threshold = pickle.load(file)
    N = pickle.load( file)
    T = pickle.load(file)
    GPMW_LCB_loss = pickle.load(file)
    GPMW_LCB_g = pickle.load(file)
    GPMW_played_actions = pickle.load(file)
    GPMW_obtained_loss = pickle.load(file)
    GPMW_cum_loss = pickle.load(file)
    GPMW_cum_constraints = pickle.load(file)
    GPMW_consumed_ernergy = pickle.load(file)
    GPMW_regrets = pickle.load(file)
    GPMW_weights = pickle.load(file)
    GPMW_history_weights = pickle.load(file)



#%% 
'''
Load the true energy consumption and temperature deviation 
of each player for all paramter combinations.
Load the true cost function of each player for all paramter 
combinations.  
'''
testing = False

seed = 8
if testing:
    with open(f"game_data_constraint_test_seed={seed}.pckl", 'rb') as file:
        true_temp_deviation = pickle.load(file)
        true_energy_hourly = pickle.load(file)
        true_room_temp = pickle.load(file)
    with open(f"game_data_cost_test_seed={seed}.pckl", 'rb') as file:
        true_cost = pickle.load(file)
else:
    with open(f"game_data_constraint0_seed={seed}.pckl", 'rb') as file:
        true_temp_deviation_0 = pickle.load(file)
        true_energy_hourly_0 = pickle.load(file)
        true_room_temp_0 = pickle.load(file)
    with open(f"game_data_constraint1_seed={seed}.pckl", 'rb') as file:
        true_temp_deviation_1 = pickle.load(file)
        true_energy_hourly_1 = pickle.load(file)
        true_room_temp_1 = pickle.load(file)
    with open(f"game_data_constraint2_seed={seed}.pckl", 'rb') as file:
        true_temp_deviation_2 = pickle.load(file)
        true_energy_hourly_2 = pickle.load(file)
        true_room_temp_2 = pickle.load(file)
    with open(f"game_data_cost0_integer.pckl", 'rb') as file:
        true_cost0 = pickle.load(file)
    with open(f"game_data_cost1_integer.pckl", 'rb') as file:
        true_cost1 = pickle.load(file)
    with open(f"game_data_cost2_integer.pckl", 'rb') as file:
        true_cost2 = pickle.load(file)
    true_cost = [true_cost0, true_cost1, true_cost2]
    true_temp_deviation = []
    true_energy_hourly = []
    true_room_temp = []
    for i in range(N):
        true_temp_deviation.append(true_temp_deviation_0[i] + true_temp_deviation_1[i] + 
                                true_temp_deviation_2[i]) 
        true_energy_hourly.append(true_energy_hourly_0[i] + true_energy_hourly_1[i] +
                                true_energy_hourly_2[i])
        true_room_temp.append(true_room_temp_0[i] + true_room_temp_1[i] +
                            true_room_temp_2[i])
        
#%%
with open('min_cost_values.pckl', 'rb') as file:
    min_cost = pickle.load(file)
    min_feasible_cost = pickle.load(file)


#%% 
########################### Plot 1 ####################################
'''
Plot the (average) room temperature and the comfort bounds of each household (player)
which are result of input parameters sampled from the learned weights.
''' 
seed = 1
np.random.seed(seed)
idx_list = np.array(list(itertools.product(range(len_P),range(len_SP_day),range(len_shift_night))))
params_list = np.array(list(itertools.product(P, SP_day, shift_night)))
len_params = len_P * len_SP_day * len_shift_night    


'''
Sample parameters from the learned weights after T rounds and generate the 
corresponding room temperatures and costs.
'''
size = 1000
# weights learned by cAdaNormalGP
sampled_actions = []
temp = [[],[],[]]
temp_dev = [[],[],[]]
# weights learned by GPMW
sampled_actions_GPMW = []
temp_GPMW = [[],[],[]]
temp_dev_GPMW = [[],[],[]]
# uniform weights
sampled_actions_uniform = []
temp_uniform = [[],[],[]]
for i in range(N):
    # weights learned by cAdaNormalGP
    indic = [1 if ai<= dev_threshold else 0 for ai in LCB_g[0][i]]
    prob = np.multiply(indic,weights[i]) / np.sum(np.multiply(indic,weights[i]))
    help = np.random.choice(len_params, size = size, p = prob)
    sampled_actions.append(list(help))
    # weights learned by GPMW
    prob = GPMW_weights[i] / np.sum(GPMW_weights[i])
    help = np.random.choice(len_params, size = size, p = prob)
    sampled_actions_GPMW.append(list(help))
    # uniform weights
    help = np.random.choice(len_params, size = size)
    sampled_actions_uniform.append(list(help))

for j in range(size):
    for i in range(N):
        # weights learned by cAdaNormalGP
        help = idx_list[sampled_actions[i][j]]
        temp[i].append(true_room_temp[i][len_SP_day*len_shift_night*help[0] +
                                         len_shift_night*help[1] +
                                         help[2]])
        temp_dev[i].append(true_temp_deviation[i][len_SP_day*len_shift_night*help[0] +
                                         len_shift_night*help[1] +
                                         help[2]])
        # weights learned by GPMW
        help = idx_list[sampled_actions_GPMW[i][j]]
        temp_GPMW[i].append(true_room_temp[i][len_SP_day*len_shift_night*help[0] +
                                         len_shift_night*help[1] +
                                         help[2]])
        temp_dev_GPMW[i].append(true_temp_deviation[i][len_SP_day*len_shift_night*help[0] +
                                         len_shift_night*help[1] +
                                         help[2]])
        # uniform weights
        help = idx_list[sampled_actions_uniform[i][j]]
        temp_uniform[i].append(true_room_temp[i][len_SP_day*len_shift_night*help[0] +
                                         len_shift_night*help[1] +
                                         help[2]])
    
   
# For testing a single sampled point       
'''
i = 1
k = 0
pseudo_std = np.array([1] * len(temp[i][k]))
SP_dev = 1.5
shift_night_played = params_list[sampled_actions[i][k]][2]
SP_day_played = params_list[sampled_actions[i][k]][1]
plot_room_temp(temp[i][k], pseudo_std, SP_night[i], SP_day_played, SP_dev, shift_day[i], shift_night_played, sim_days)
'''

# weights learned by cAdaNormalGP
temp_avg = [[],[],[]]
temp_std = [[],[],[]]
shift_night_avg = [[],[],[]]
SP_day_avg = [[],[],[]]
# weights learned by cAdaNormalGP
temp_GPMW_avg = [[],[],[]]
temp_GPMW_std = [[],[],[]]
shift_night_GPMW_avg = [[],[],[]]
SP_day_GPMW_avg = [[],[],[]]
# uniform weights
temp_uniform_avg = [[],[],[]]
temp_uniform_std = [[],[],[]]
shift_night_uniform_avg = [[],[],[]]
SP_day_uniform_avg = [[],[],[]]
for i in range(N):
    # weights learned by cAdaNormalGP
    temp_avg[i].append(np.mean(np.array(temp[i]), axis=0))
    temp_std[i].append(np.std(np.array(temp[i]), axis=0))
    help1 = [params_list[sampled_actions[i][k]][2] for k in range(size)]
    shift_night_avg[i].append(np.mean(help1, axis=0))
    help2 = [params_list[sampled_actions[i][k]][1] for k in range(size)]
    SP_day_avg[i].append(np.mean(help2, axis=0))
    # weights learned by cAdaNormalGP
    temp_GPMW_avg[i].append(np.mean(np.array(temp_GPMW[i]), axis=0))
    temp_GPMW_std[i].append(np.std(np.array(temp_GPMW[i]), axis=0))
    help1 = [params_list[sampled_actions_GPMW[i][k]][2] for k in range(size)]
    shift_night_GPMW_avg[i].append(np.mean(help1, axis=0))
    help2 = [params_list[sampled_actions_GPMW[i][k]][1] for k in range(size)]
    SP_day_GPMW_avg[i].append(np.mean(help2, axis=0))
    # uniform weights
    temp_uniform_avg[i].append(np.mean(np.array(temp_uniform[i]), axis=0))
    temp_uniform_std[i].append(np.std(np.array(temp_uniform[i]), axis=0))
    help1 = [params_list[sampled_actions_uniform[i][k]][2] for k in range(size)]
    shift_night_uniform_avg[i].append(np.mean(help1, axis=0))
    help2 = [params_list[sampled_actions_uniform[i][k]][1] for k in range(size)]
    SP_day_uniform_avg[i].append(np.mean(help2, axis=0))

#%%
# cAdaNormalGP
i = 1
SP_dev = 1.5
plot_room_temp(temp_avg[i][0], temp_std[i][0], SP_night[i], SP_day_avg[i][0], SP_dev, shift_day[i], shift_night_avg[i][0], sim_days)
#plt.savefig(f'temp_deviation_cAdaNormalGP_player_{i}.png', bbox_inches='tight', pad_inches=0)

#%%
# GPMW
plot_room_temp(temp_GPMW_avg[i][0], temp_GPMW_std[i][0], SP_night[i], SP_day_GPMW_avg[i][0], SP_dev, shift_day[i], shift_night_GPMW_avg[i][0], sim_days)
#plt.savefig(f'temp_deviation_GPMW_player_{i}.png',bbox_inches='tight', pad_inches=0)

#%%
# cAdaNormalGP and GPMW in one plot
i = 1
SP_dev = 1.5

plot_room_temp_combined(temp_avg[i][0], temp_GPMW_avg[i][0], temp_std[i][0],temp_GPMW_std[i][0],
                SP_night[i], SP_day_avg[i][0], SP_day_GPMW_avg[i][0], SP_dev,
                  shift_day[i], shift_night_avg[i][0], shift_night_GPMW_avg[i][0], sim_days)


#plt.savefig(f'temp_deviation_combined_2.pdf',bbox_inches='tight', pad_inches=0)


#%% 
################## Plot 2 ############################
'''
Plot the (average) costs of each household (player)
which are result of input parameters sampled from the learned weights at times 
t=1,...,T.
''' 
seed = 1
np.random.seed(seed)
idx_list = np.array(list(itertools.product(range(len_P),range(len_SP_day),range(len_shift_night))))
params_list = np.array(list(itertools.product(P, SP_day, shift_night)))
len_params = len_P * len_SP_day * len_shift_night    

'''
Sample parameters from the learned weights  after T rounds and generate the 
corresponding room temperatures and costs.
'''
size = 1000
# weights learned by cAdaNormalGP
cost_avg = []
cost_std = []
# weights learned by GPMW
cost_GPMW_avg = []
cost_GPMW_std = []
# uniform weights
cost_uniform_avg = []
cost_uniform_std = []
for t in range(T):
    # weights learned by cAdaNormalGP
    cost_avg.append( [None]*N )
    cost_std.append( [None]*N)
    sampled_actions = []
    cost = [[],[],[]]
    # weights learned by GPMW
    cost_GPMW_avg.append( [None]*N )
    cost_GPMW_std.append( [None]*N)
    sampled_actions_GPMW = []
    cost_GPMW = [[],[],[]]
    # uniform weights
    cost_uniform_avg.append( [None]*N )
    cost_uniform_std.append( [None]*N)
    sampled_actions_uniform = []
    cost_uniform = [[],[],[]]
    # weights learned by cAdaNormalGP
    for i in range(N):
        #prob = history_weights[0][t][i]/ np.sum(history_weights[0][t][i])
        indic = [1 if ai<= dev_threshold else 0 for ai in history_LCB_g[0][t][i]]
        prob = np.multiply(indic,history_weights[0][t][i]) / np.sum(np.multiply(indic,history_weights[0][t][i]))
        help = np.random.choice(len_params, size = size, p = prob)
        sampled_actions.append(list(help))
    for j in range(size):
        help0 = idx_list[sampled_actions[0][j]]
        help1 = idx_list[sampled_actions[1][j]]
        help2 = idx_list[sampled_actions[2][j]]
        for i in range(N):
            cost[i].append(true_cost[i][
                                        len_P**2*len_SP_day**3*len_shift_night**3*help0[0] + len_P**2*len_SP_day**2*len_shift_night**3*help0[1] + len_P**2*len_SP_day**2*len_shift_night**2*help0[2] +
                                        len_P*len_SP_day**2*len_shift_night**2*help1[0] + len_P*len_SP_day*len_shift_night**2*help1[1] + len_P*len_SP_day*len_shift_night*help1[2]+
                                        len_SP_day*len_shift_night*help2[0] + len_shift_night*help2[1] + help2[2]
                        ]
                        )

    for i in range(N):
        cost_avg[t][i] = np.mean(np.array(cost[i]), axis=0)
        cost_std[t][i] = np.std(np.array(cost[i]), axis=0)
    # weights learned by GPMW
    for i in range(N):
        prob = GPMW_history_weights[0][t][i]/ np.sum(GPMW_history_weights[0][t][i])
        help = np.random.choice(len_params, size = size, p = prob)
        sampled_actions_GPMW.append(list(help))
    for j in range(size):
        help0 = idx_list[sampled_actions_GPMW[0][j]]
        help1 = idx_list[sampled_actions_GPMW[1][j]]
        help2 = idx_list[sampled_actions_GPMW[2][j]]
        for i in range(N):
            cost_GPMW[i].append(true_cost[i][
                                        len_P**2*len_SP_day**3*len_shift_night**3*help0[0] + len_P**2*len_SP_day**2*len_shift_night**3*help0[1] + len_P**2*len_SP_day**2*len_shift_night**2*help0[2] +
                                        len_P*len_SP_day**2*len_shift_night**2*help1[0] + len_P*len_SP_day*len_shift_night**2*help1[1] + len_P*len_SP_day*len_shift_night*help1[2]+
                                        len_SP_day*len_shift_night*help2[0] + len_shift_night*help2[1] + help2[2]
                        ]
                        )

    for i in range(N):
        cost_GPMW_avg[t][i] = np.mean(np.array(cost_GPMW[i]), axis=0)
        cost_GPMW_std[t][i] = np.std(np.array(cost_GPMW[i]), axis=0)
    # uniform weights
    for i in range(N):
        help = np.random.choice(len_params, size = size)
        sampled_actions_uniform.append(list(help))
    for j in range(size):
        help0 = idx_list[sampled_actions_uniform[0][j]]
        help1 = idx_list[sampled_actions_uniform[1][j]]
        help2 = idx_list[sampled_actions_uniform[2][j]]
        for i in range(N):
            cost_uniform[i].append(true_cost[i][
                                        len_P**2*len_SP_day**3*len_shift_night**3*help0[0] + len_P**2*len_SP_day**2*len_shift_night**3*help0[1] + len_P**2*len_SP_day**2*len_shift_night**2*help0[2] +
                                        len_P*len_SP_day**2*len_shift_night**2*help1[0] + len_P*len_SP_day*len_shift_night**2*help1[1] + len_P*len_SP_day*len_shift_night*help1[2]+
                                        len_SP_day*len_shift_night*help2[0] + len_shift_night*help2[1] + help2[2]
                        ]
                        )

    for i in range(N):
        cost_uniform_avg[t][i] = np.mean(np.array(cost_uniform[i]), axis=0)
        cost_uniform_std[t][i] = np.std(np.array(cost_uniform[i]), axis=0)
    
        
#%%
i = 1
plot_cost(i,cost_avg, cost_GPMW_avg, cost_uniform_avg, cost_std,
           cost_GPMW_std, cost_uniform_std, min_cost, min_feasible_cost)


#plt.savefig(f'cost_player_{i}.pdf')

