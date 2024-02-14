#%%
import numpy as np
import itertools
import pickle
from funcs_thermal_control_cost import *
#%%
#################### Generate data #################################
'''
Generate the roomtemperature, the temperature deviation, the energy consumption
and the energy cost for each household for each combination of action profiles
in the action space.
We consided a discretized action space A_i = P x SP_day x ST_night, where 
P is the proportianl gain of the P controller, SP_day is the setpoint temperature during
the day and ST_night=shift_night is the shifting time from the setpoint temperature
during the to the setpoint temperature during the night. The discretized action space
of each building/ household is given by:

P = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0]
SP_day = [16,17,18,19,20,21]
shift_night = [16, 17, 18, 19, 20]

'''
seed = 8
np.random.seed(seed)

# weather data
weather = "CH_BS_Basel"
# number of agents
N = 3

# Initialize player
Player = list(range(N))
SP_night = [21, 22, 19]
shift_day = [8, 6, 7] # equivalent to nighttime_end
for i in range(N):
    Player[i] = Household(SP_night[i], shift_day[i])

# Action set of each player
testing = False
if testing:
    P = [0.2, 0.5, 0.8, 1.2]
    SP_day = [17, 18, 19]
    shift_night = [16,18,20]
else:
    P = [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0]
    SP_day = [16,17,18,19,20,21]
    shift_night = [16, 17, 18, 19, 20]

'''
P - Control gain of P controller, type: list
SP_day - set point temperature during the day, type: list
shift_night - night time shift, type: list
'''

len_P = len(P)
#len_P = 12
len_SP_day = len(SP_day)
len_shift_night = len(shift_night)
sim_days = 2


#%%
# Generate the true costs and constraints of each agent
count = 0
true_temp_deviation = [[],[],[]]
true_energy_hourly = [[],[],[]]
true_room_temp = [[],[],[]]
#input = np.array(list(itertools.product(range(len_P),range(len_SP_day), range(len_shift_night))))
input = np.array(list(itertools.product(P, SP_day, shift_night)))
for idx in input:
    print(idx)
    for i in range(N):
        params = [idx[0], idx[1], idx[2]]
        avg_energy_hourly, tot_energy_hourly, temp_list, avg_dev = Player[i].get_ApartTherm_kpis(weather=weather, sim_days=sim_days, params=params)
        true_temp_deviation[i].append(max(avg_dev))
        true_energy_hourly[i].append(tot_energy_hourly)
        true_room_temp[i].append(temp_list)
    count = count +1
        
with open(f"game_data_max_constraint_seed={seed}.pckl", 'wb') as file:
    pickle.dump(true_temp_deviation, file)
    pickle.dump(true_energy_hourly, file)
    pickle.dump(true_room_temp, file)



#%%
with open(f"game_data_constraint_seed={seed}.pckl", 'rb') as file:
    true_temp_deviation = pickle.load(file)
    true_energy_hourly = pickle.load(file)
#%%
true_cost = [[],[],[]]
input = np.array(list(itertools.product(range(len_P),range(len_SP_day), range(len_shift_night))))

for idx0 in range(len(input)):
    print(f'round={idx0}')
    for idx1 in range(len(input)):
        for idx2 in range(len(input)):
            energy_hourly = [true_energy_hourly[0][idx0], true_energy_hourly[1][idx1], true_energy_hourly[2][idx2]]
            for i in range(N):
                true_cost[i].append(Player[i].get_ApartElectricity_cost(num_agents=N, agent=i, sim_days=sim_days, energy_hourly=energy_hourly))

with open(f"game_data_cost_seed={seed}.pckl", 'wb') as file:
    pickle.dump(true_cost, file)










#%%

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
with open(f"game_data_max_constraint_seed={seed}.pckl", 'wb') as file:
            pickle.dump(true_temp_deviation, file)
            pickle.dump(true_energy_hourly, file)
            pickle.dump(true_room_temp, file)


with open(f"game_data_cost_0_seed={seed}.pckl", 'rb') as file:
    true_cost_0 = pickle.load(file)
    

with open(f"game_data_cost_seed={seed}.pckl", 'rb') as file:
    true_cost_1 = pickle.load(file)

true_cost = []
for i in range(N):
    true_cost.append(true_cost_0[i] + true_cost_1[i] ) 


with open(f"game_data_true_cost_seed={seed}.pckl", 'wb') as file:
    pickle.dump(true_cost, file)
            