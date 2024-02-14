#%%
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np

#%%

def plot_room_temp(list_temps, list_std, SP_night, SP_day, SP_dev, shift_day, shift_night, sim_days):

    '''
    Plot the room temperature, the setpoint temperature, the comfort range
    during the night and the switching times for a single building.
    '''
    horizon = sim_days * 24
    fig = plt.figure(figsize=(6,4))
    plt.plot(range(horizon), list_temps, color='black', marker = '*', markevery = 2, markersize = 7, label = 'Room temperature')
    plt.fill_between(range(horizon), list_temps - list_std, list_temps + list_std, alpha=0.05,color='b')  
    # SP
    plt.hlines(SP_night, xmin=0, xmax=shift_day, colors='r', linestyles='solid', label='Setpoint temperature')
    plt.hlines(SP_day, xmin=shift_day, xmax=shift_night, colors='r', linestyles='solid')
    plt.hlines(SP_night, xmin=shift_night, xmax=24+shift_day, colors='r', linestyles='solid')
    plt.hlines(SP_day, xmin=24+shift_day, xmax=24+shift_night, colors='r', linestyles='solid')
    plt.hlines(SP_night, xmin=24+shift_night, xmax=24+24, colors='r', linestyles='solid')
    # SP + dev
    plt.hlines(SP_night + SP_dev, xmin=0, xmax=shift_day, colors='r', linestyles='dashed', label='Comfort zone')
    #plt.hlines(SP_day + SP_dev, xmin=shift_day, xmax=shift_night, colors='r', linestyles='dashed')
    plt.hlines(SP_night + SP_dev, xmin=shift_night, xmax=24+shift_day, colors='r', linestyles='dashed')
    #plt.hlines(SP_day + SP_dev, xmin=24+shift_day, xmax=24+shift_night, colors='r', linestyles='dashed')
    plt.hlines(SP_night + SP_dev, xmin=24+shift_night, xmax=24+24, colors='r', linestyles='dashed')
    # SP -dev
    plt.hlines(SP_night - SP_dev, xmin=0, xmax=shift_day, colors='r', linestyles='dashed')
    #plt.hlines(SP_day - SP_dev, xmin=shift_day, xmax=shift_night, colors='r', linestyles='dashed')
    plt.hlines(SP_night - SP_dev, xmin=shift_night, xmax=24+shift_day, colors='r', linestyles='dashed')
    #plt.hlines(SP_day - SP_dev, xmin=24+shift_day, xmax=24+shift_night, colors='r', linestyles='dashed')
    plt.hlines(SP_night - SP_dev, xmin=24+shift_night, xmax=24+24, colors='r', linestyles='dashed')
    
    plt.vlines(shift_day, ymin=0, ymax=24, colors='b', linestyles='dashed', label='Day to nighttime shift')
    plt.vlines(shift_night, ymin=0, ymax=24, colors='b', linestyles='dashed')
    plt.vlines(24+shift_day, ymin=0, ymax=24, colors='b', linestyles='dashed')
    plt.vlines(24+shift_night, ymin=0, ymax=24, colors='b', linestyles='dashed')


    x = [0, shift_day, shift_night, 24+shift_day, 24+shift_night, 24+24] 
    y = [0,0,0,0,0,0]
    text = ['Night', 'Day', 'Night', 'Day', 'Night', 'Day']
    plt.step(x, y, 'white', linewidth=0.0001, where='post')
    text_x = [0, shift_day+1, shift_night+1, 24+shift_day+1, 24+shift_night+1, 24+24+1] 
    text_y = [12]*len(x)

    for t, tx, ty in zip(text[:-1], text_x, text_y):
        plt.text(tx, ty, t, fontsize=14)

    plt.legend(prop={'size': 14}, loc='lower left',facecolor='white', framealpha=1)
    plt.xlabel('Hours', fontsize=18)
    plt.xticks(fontsize=18)
    plt.ylabel('Room temperature', fontsize=18)
    plt.yticks(fontsize=16)
    plt.xlim = (0,48)
    fig.tight_layout()
    fig.canvas.draw()



def plot_room_temp_combined(list_temps, list_temps_GPMW, list_std, 
                   list_std_GPMW, SP_night, SP_day, SP_day_GPMW, 
                   SP_dev, shift_day, shift_night, shift_night_GPMW, sim_days
                   ):

    '''
    Plot the room temperature, the setpoint temperature, the comfort range
    during the night and the switching times for a single building resulting from
    two different learning algorithms.
    '''

    horizon = sim_days * 24
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6)) 
    
    # Plot 1
    axes[0].plot(range(horizon), list_temps, color='black', marker = '*', markevery = 2, markersize = 7, label = 'Room temperature')
    axes[0].fill_between(range(horizon), list_temps - list_std, list_temps + list_std, alpha=0.05,color='b')  
    # SP
    axes[0].hlines(SP_night, xmin=0, xmax=shift_day, colors='r', linestyles='solid', label='Setpoint temperature')
    axes[0].hlines(SP_day, xmin=shift_day, xmax=shift_night, colors='r', linestyles='solid')
    axes[0].hlines(SP_night, xmin=shift_night, xmax=24+shift_day, colors='r', linestyles='solid')
    axes[0].hlines(SP_day, xmin=24+shift_day, xmax=24+shift_night, colors='r', linestyles='solid')
    axes[0].hlines(SP_night, xmin=24+shift_night, xmax=24+24, colors='r', linestyles='solid')
    # SP + dev
    axes[0].hlines(SP_night + SP_dev, xmin=0, xmax=shift_day, colors='r', linestyles='dashed', label='Comfort zone')
    #plt.hlines(SP_day + SP_dev, xmin=shift_day, xmax=shift_night, colors='r', linestyles='dashed')
    axes[0].hlines(SP_night + SP_dev, xmin=shift_night, xmax=24+shift_day, colors='r', linestyles='dashed')
    #plt.hlines(SP_day + SP_dev, xmin=24+shift_day, xmax=24+shift_night, colors='r', linestyles='dashed')
    axes[0].hlines(SP_night + SP_dev, xmin=24+shift_night, xmax=24+24, colors='r', linestyles='dashed')
    # SP -dev
    axes[0].hlines(SP_night - SP_dev, xmin=0, xmax=shift_day, colors='r', linestyles='dashed')
    #plt.hlines(SP_day - SP_dev, xmin=shift_day, xmax=shift_night, colors='r', linestyles='dashed')
    axes[0].hlines(SP_night - SP_dev, xmin=shift_night, xmax=24+shift_day, colors='r', linestyles='dashed')
    #plt.hlines(SP_day - SP_dev, xmin=24+shift_day, xmax=24+shift_night, colors='r', linestyles='dashed')
    axes[0].hlines(SP_night - SP_dev, xmin=24+shift_night, xmax=24+24, colors='r', linestyles='dashed')
    
    axes[0].vlines(shift_day, ymin=0, ymax=24, colors='b', linestyles='dashed', label='Day to nighttime shift')
    axes[0].vlines(shift_night, ymin=0, ymax=24, colors='b', linestyles='dashed')
    axes[0].vlines(24+shift_day, ymin=0, ymax=24, colors='b', linestyles='dashed')
    axes[0].vlines(24+shift_night, ymin=0, ymax=24, colors='b', linestyles='dashed')

    x = [0, shift_day, shift_night, 24+shift_day, 24+shift_night, 24+24] 
    y = [0,0,0,0,0,0]
    text = ['Night', 'Day', 'Night', 'Day', 'Night', 'Day']
    axes[0].step(x, y, 'white', linewidth=0.0001, where='post')
    text_x = [0, shift_day+1, shift_night+1, 24+shift_day+1, 24+shift_night+1, 24+24+1] 
    text_y = [0.5]*len(x)

    for t, tx, ty in zip(text[:-1], text_x, text_y):
        axes[0].text(tx, ty, t, fontsize=14)

    axes[0].set_ylabel('Room temperature', fontsize=18)
    

    # Plot 2
    axes[1].plot(range(horizon), list_temps_GPMW, color='black', marker = '*', markevery = 2, markersize = 7, label = 'Room temperature')
    axes[1].fill_between(range(horizon), list_temps_GPMW - list_std_GPMW, list_temps_GPMW + list_std_GPMW, alpha=0.05,color='b')  
    # SP
    axes[1].hlines(SP_night, xmin=0, xmax=shift_day, colors='r', linestyles='solid', label='Setpoint temperature')
    axes[1].hlines(SP_day_GPMW, xmin=shift_day, xmax=shift_night_GPMW, colors='r', linestyles='solid')
    axes[1].hlines(SP_night, xmin=shift_night_GPMW, xmax=24+shift_day, colors='r', linestyles='solid')
    axes[1].hlines(SP_day_GPMW, xmin=24+shift_day, xmax=24+shift_night_GPMW, colors='r', linestyles='solid')
    axes[1].hlines(SP_night, xmin=24+shift_night_GPMW, xmax=24+24, colors='r', linestyles='solid')
    # SP + dev
    axes[1].hlines(SP_night + SP_dev, xmin=0, xmax=shift_day, colors='r', linestyles='dashed', label='Comfort zone')
    #plt.hlines(SP_day + SP_dev, xmin=shift_day, xmax=shift_night, colors='r', linestyles='dashed')
    axes[1].hlines(SP_night + SP_dev, xmin=shift_night_GPMW, xmax=24+shift_day, colors='r', linestyles='dashed')
    #plt.hlines(SP_day + SP_dev, xmin=24+shift_day, xmax=24+shift_night, colors='r', linestyles='dashed')
    axes[1].hlines(SP_night + SP_dev, xmin=24+shift_night_GPMW, xmax=24+24, colors='r', linestyles='dashed')
    # SP -dev
    axes[1].hlines(SP_night - SP_dev, xmin=0, xmax=shift_day, colors='r', linestyles='dashed')
    #plt.hlines(SP_day - SP_dev, xmin=shift_day, xmax=shift_night, colors='r', linestyles='dashed')
    axes[1].hlines(SP_night - SP_dev, xmin=shift_night_GPMW, xmax=24+shift_day, colors='r', linestyles='dashed')
    #plt.hlines(SP_day - SP_dev, xmin=24+shift_day, xmax=24+shift_night, colors='r', linestyles='dashed')
    axes[1].hlines(SP_night - SP_dev, xmin=24+shift_night_GPMW, xmax=24+24, colors='r', linestyles='dashed')
    
    axes[1].vlines(shift_day, ymin=0, ymax=24, colors='b', linestyles='dashed', label='Day to nighttime shift')
    axes[1].vlines(shift_night_GPMW, ymin=0, ymax=24, colors='b', linestyles='dashed')
    axes[1].vlines(24+shift_day, ymin=0, ymax=24, colors='b', linestyles='dashed')
    axes[1].vlines(24+shift_night_GPMW, ymin=0, ymax=24, colors='b', linestyles='dashed')


    x = [0, shift_day, shift_night_GPMW, 24+shift_day, 24+shift_night_GPMW, 24+24] 
    y = [0,0,0,0,0,0]
    text = ['Night', 'Day', '', 'Day', 'Night', 'Day']
    axes[0].step(x, y, 'white', linewidth=0.0001, where='post')
    text_x = [0, shift_day+1, shift_night_GPMW+1, 24+shift_day+1, 24+shift_night_GPMW+1, 24+24+1] 
    text_y = [0.5]*len(x)

    for t, tx, ty in zip(text[:-1], text_x, text_y):
        axes[1].text(tx, ty, t, fontsize=14)

    axes[1].legend(prop={'size': 12}, loc='lower left',facecolor='white', framealpha=1)
    axes[1].set_xlabel('Hours', fontsize=18)
    axes[1].set_ylabel('Room temperature', fontsize=18)
    axes[1].set_xlim = ([0,48])
    fig.tight_layout()
    fig.canvas.draw()




def plot_cost(i, cost_avg, cost_avg_GPMW, cost_avg_uniform, cost_std,
              cost_std_GPMW, cost_std_uniform, min_cost, min_feasible_cost):

    '''
    Plot the energy cost function as function of the number of rounds T for
    a single household when the household follows c.AdaNormalGP, GPMw or plays
    uniformly at random. The minimal and the minimal feasible cost achievable, 
    found by exhaustively searching the action space, are also plotted. 
    '''
    N = len(cost_avg[0])
    T = len(cost_avg)
    color = ['blue', 'red', 'green']
    marker = ['*', '+', 'o']
    
    fig = plt.figure(figsize=(6, 3.5)) #(7, 4.5)
    # cAdaNormalGP
    cost_avg_player = np.array([cost_avg[t][i] for t in range(T)])
    cost_std_player = np.array([cost_std[t][i] for t in range(T)])
    plt.plot(range(T), cost_avg_player, color='blue', marker = marker[0], markevery = 50, markersize = 7, label = 'cAdaNormalGP' )
    plt.fill_between(range(T), cost_avg_player - cost_std_player, cost_avg_player + cost_std_player, alpha=0.05,color='black')  
    plt.hlines(min_cost[i], xmin=0, xmax=T, colors='red', linestyles='dashed', label= 'min cost')
    plt.hlines(min_feasible_cost[i], xmin=0, xmax=T, colors='green', linestyles='dashed', label = 'min feasible cost')
    # GPMW
    cost_avg_player = np.array([cost_avg_GPMW[t][i] for t in range(T)])
    cost_std_player = np.array([cost_std_GPMW[t][i] for t in range(T)])
    plt.plot(range(T), cost_avg_player, color='purple', marker = marker[1], markevery = 50, markersize = 7, label='GPMW')
    plt.fill_between(range(T), cost_avg_player - cost_std_player, cost_avg_player + cost_std_player, alpha=0.05,color='black')  
    # uniform
    cost_avg_player = np.array([cost_avg_uniform[t][i] for t in range(T)])
    cost_std_player = np.array([cost_std_uniform[t][i] for t in range(T)])
    plt.plot(range(T), cost_avg_player, color='black', marker = marker[2], markevery = 50, markersize = 7, label ='uniform' )
    #plt.fill_between(range(T), cost_avg_player - cost_std_player, cost_avg_player + cost_std_player, alpha=0.05,color='black')  

    
    
    plt.legend(prop={'size': 10}, loc='upper right')
    plt.xlabel('T', fontsize=18)
    plt.ylabel('Cost', fontsize=18)
    plt.ylim(1400, 2400)
    plt.xlim(0,500)
    fig.tight_layout()
    #plt.rcParams.update({'font.size': 16})
    fig.canvas.draw()
    
