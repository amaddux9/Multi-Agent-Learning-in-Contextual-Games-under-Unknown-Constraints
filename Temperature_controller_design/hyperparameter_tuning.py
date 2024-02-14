#%%
import numpy as np
import GPy
import pickle 
from funcs_thermal_control_cost import *
from IPython.display import display

#%%
'''
Tune hyperparameters for the kernel of each player for 
their cost and the constraint function
'''
seed = 8
np.random.seed(seed)


# weather data
weather = "CH_BS_Basel"

# number of agents
N = 3

# Initialize player
sim_days = 2
Player = list(range(N))
SP_night = [21, 22, 19]
shift_day = [8, 6, 7] # equivalent to nighttime_end
for i in range(N):
    Player[i] = Household(SP_night[i], shift_day[i])

# for GPy
sigma1 = 0.1 

sample_size = 120
P = np.random.uniform(0.2,2.5, sample_size*N)
SP_day = np.random.choice([15,16,17,18,19,20,21,22], sample_size*N)
shift_night = np.random.choice([15,16, 17, 18, 19, 20,21], sample_size*N)

'''
Compute temperature deviation and hourly energy consumption for the
three households for the randomly drawn parameters P, SP_day, shift_night
'''
true_temp_deviation = [[], [], []]
true_energy_hourly = [[], [], []]
params_g = [[],[],[]]
for i in range(N):
    for idx in range(sample_size):
        params = [P[idx+sample_size*i], SP_day[idx+sample_size*i], shift_night[idx+sample_size*i]]
        params_g[i].append(params)
        avg_energy_hourly, tot_energy_hourly, out_list, avg_dev = Player[i].get_ApartTherm_kpis(weather=weather, sim_days=sim_days, params=params)
        true_temp_deviation[i].append(max(avg_dev))
        true_energy_hourly[i].append(tot_energy_hourly)

'''
Compute the cost from the aggragte energy consumption for the
three households for the randomly drawn parameters P, SP_day, shift_night
'''
true_cost = [[],[],[]]
params_l = []
for idx in range(sample_size):
    params_l.append(true_energy_hourly[0][idx] + true_energy_hourly[1][idx] + true_energy_hourly[2][idx])
    energy_hourly = [true_energy_hourly[0][idx], true_energy_hourly[1][idx], true_energy_hourly[2][idx]]
    for i in range(N):
        true_cost[i].append(Player[i].get_ApartElectricity_cost(num_agents=N, agent=i, sim_days=sim_days, energy_hourly=energy_hourly))


with open(f'hyperparamter_tuning.pckl', 'wb') as file:
    pickle.dump(sample_size, file)
    pickle.dump(P, file)
    pickle.dump(SP_day, file)
    pickle.dump(shift_night, file)
    pickle.dump(params_g , file)
    pickle.dump(true_temp_deviation , file)
    pickle.dump(params_l, file)
    pickle.dump(true_cost, file)

#%%
with open(f'hyperparamter_tuning.pckl', 'rb') as file:
    sample_size = pickle.load(file)
    P = pickle.load(file)
    SP_day = pickle.load(file)
    shift_night = pickle.load(file)
    params_g = pickle.load( file)
    true_temp_deviation = pickle.load( file)
    params_l = pickle.load(file)
    true_cost = pickle.load( file)
#%%
'''
Hyperparameter tuning for the constraint function, i.e. the 
temperature deviation
'''
lengthscale_g = []
for i in range(N):
    X_g = np.array(params_g[i])
    helper_g = np.array(true_temp_deviation[i])
    Y_g = helper_g.reshape(len(helper_g), 1)
    model_g = GPy.models.GPRegression(X_g,Y_g, GPy.kern.RBF(input_dim=3, active_dims=range(3), ARD=True))
    model_g.Gaussian_noise.fix(sigma1**2)
    display(model_g)

    model_g.optimize_restarts(num_restarts = 10)
    display(model_g)
    lengthscale_g.append(model_g.kern.lengthscale.values)



#%%
'''
Hyperparameter tuning for the cost function, i.e. the price of 
the consumed energy with polynomial kernel
'''
bias = []
for i in range(N):
    # take polynomial kernel
    X_l = np.array(params_l)
    helper_l = np.array(true_cost[i])
    Y_l = helper_l.reshape(len(helper_l), 1)
    model_l = GPy.models.GPRegression(X_l,Y_l, GPy.kern.Poly(input_dim=sim_days*24*N, order=2))
    model_l.Gaussian_noise.fix(sigma1**2)
    display(model_l)

    model_l.optimize_restarts(num_restarts = 8)
    display(model_l)
    bias.append(model_l.kern.bias)


