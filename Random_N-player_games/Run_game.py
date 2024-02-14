#%%
"""
Look at Piers code for contextual games. He does 
hyperparameter tuning. I don't think I need to because I 
generated my data by sampling from a GP and therefore know 
the true hyperparameters.
"""

import numpy as np
import GPy
import matplotlib.pyplot as plt
import sys
import pickle
import itertools
from my_aux_functions import *

plt.close('all')
#%%
'''
Generate the reward and constraint function
'''
# Generate reward function
def Generate_r(N,K,Z):

    # Define kernel for action profile and contexts
    # RBF Kernel
    l = 2.0
    k1 = GPy.kern.RBF(input_dim=N, variance=1., active_dims=list(range(N)), lengthscale=l, ARD=True)
    l_context = 0.5
    k2 = GPy.kern.RBF(input_dim=1, variance=1., active_dims=[N], lengthscale=l_context, ARD=True)
    k_prod = k1 * k2

    # Compute kernel covariance matrix
    input = np.array(list(itertools.product(range(K),range(K),range(K),range(Z))))
    Cov = k_prod.K(input,input)
    
    # Sample from GP
    Mu = 0*np.ones(K**N*Z)
    Realiz = np.random.multivariate_normal(Mu, Cov, size=1)
    Realiz = np.transpose(Realiz)

    # Predict given the GP sample
    model = GPy.models.GPRegression(input,Realiz,k_prod)
    mu, var = model.predict(input)
    mu = mu.flatten()
    return mu, input

# Generate constraint function
def Generate_g(K):
    
    # Define kernel for action
    l_g = 0.5
    k = GPy.kern.RBF(input_dim=1, variance=1., active_dims=[0], lengthscale=l_g)
    
    # Compute kernel covariance matrix
    input = np.array([[a] for a in range(K)])    
    Cov = k.K(input, input)

    # Sample from GP                
    Mu = 0*np.ones(K)
    Realiz = np.random.multivariate_normal(Mu, Cov, size = 1)
    Realiz = np.transpose(Realiz)

    # Predict given the GP sample
    model = GPy.models.GPRegression(input,Realiz,k)
    mu, var = model.predict(input)
    mu = mu.flatten()
    return mu

#%%
''' 
Repeated contextual game with unknown constraints (N players use no-regret algorithm and learn constraints)
'''
class GameData:
     def __init__(self, N,T,K,r,g,contexts):
        self.Played_actions = []
        self.Mixed_strategies =  []
        self.Obtained_payoffs = [] 
        self.Cum_payoffs =  []
        self.Obtained_constraints = []
        self.Cum_constraints = [[0]*N]
        self.Regrets =  []
        self.Policies = []
        self.N = N
        self.time = T
        self.arms = K
        self.true_reward = r
        self.true_constraints = g 
        self.contexts = contexts
        # might have to update this list

def RunGame(N,K,T, r, g, types , sigma0, sigma1, contexts,rev_contexts,version):
    # contexts is the set of contexts
    # rev_contexts are the contexts that are revealed from t=1,...,T
    
    noises_r = np.random.normal(0,sigma0,T)
    noises_g = np.random.normal(0,sigma1,T)

    Game_data = GameData(N,T,K,r,g,contexts)
    Game_data.rev_contexts = rev_contexts
    Game_data.version = version
    Game_data.UCB_vectors_r = []
    Game_data.UCB_vectors_g = []
    Game_data.LCB_vectors_g = []

    
    # RBF Kernel
    # reward lies in RKHS with kernel k_prod
    l = 2.0
    k1 = GPy.kern.RBF(input_dim=N, variance=1., active_dims=list(range(N)), lengthscale=l, ARD=True)
    l_context = 0.5
    k2 = GPy.kern.RBF(input_dim=1, variance=1., active_dims=[N], lengthscale=l_context, ARD=True)
    k_prod = k1 * k2
    

    # constraint lies in RKHS with kernel k_g
    l_g = 0.5
    k_g = GPy.kern.RBF(input_dim=1, variance=1., active_dims=[0], lengthscale=l_g)
    
    Player = list(range(N))  #list of all players 
    min_payoff = []
    payoffs_range = []
    for i in range(N):
        min_payoff.append( np.array(r[i].min()))
        payoffs_range.append( np.array(r[i].max() - r[i].min()) )
        Game_data.Cum_payoffs.append( np.zeros((len(contexts),K)) )  
        Game_data.Policies.append(np.zeros(len(contexts)))

        if types[i] == 'random':
            Player[i] = Player_random(K,i)
        if types[i] == 'c_random': # constrained random player
            Player[i] = Player_c_random(K,k_g)
        if types[i] == "GPMW": 
            Player[i] = Player_GPMW(K,T,i,min_payoff[i], payoffs_range[i], k1) # use kernel k1 because GPMW ignores contexts
        if types[i] == 'zGPMW':
            Player[i] = Player_zGPMW(K, T,i, min_payoff[i], payoffs_range[i], k1, k2, version,rev_contexts[0])
        if types[i] == "cGPMW": 
            Player[i] = Player_cGPMW(K,T,i,min_payoff[i], payoffs_range[i], k1, k_g) # constrained GPMW uses MW update but learns constraints in addition, use kernel k1 because cGPMW ignores contexts
        if types[i] == "cAdaNormalGP": #constrained AdaNormalGP
            Player[i] = Player_cAdaNormalGP(K,i,min_payoff[i], payoffs_range[i], k1, k_g)
        if types[i] == "czAdaNormalGP": #constrained contextual AdaNormalGP
            Player[i] = Player_czAdaNormalGP(K,i,min_payoff[i], payoffs_range[i], k1, k2, k_g, version, rev_contexts[0])

    for t in range(T):

        if (t % 50 == 0):
            print('t= ' + str(t))

        Game_data.UCB_vectors_g.append( [None]*N)
        Game_data.LCB_vectors_g.append( [None]*N)
        Game_data.Played_actions.append(  [None]*N )

        for i in range(N):
            # Feasibility Check
            if (types[i] == 'c_random') or (types[i] == 'cGPMW') or (types[i] == 'cAdaNormalGP') or (types[i] == 'czAdaNormalGP'):
                if len([ai for ai in Player[i].lcb_constraint_est if (ai<= 0)]) == 0:
                    sys.exit("No feasible actions available.")  
                Game_data.UCB_vectors_g[t][i] = Player[i].ucb_constraint_est.tolist()
                Game_data.LCB_vectors_g[t][i] = Player[i].lcb_constraint_est.tolist()

            # Played action
            if (types[i] == 'zGPMW') or (types[i] == 'czAdaNormalGP'):
                Game_data.Played_actions[t][i] = Player[i].sample_action(rev_contexts[t])
            else:
                Game_data.Played_actions[t][i] =  Player[i].sample_action()
                
        
        Game_data.Obtained_payoffs.append([None]*N)       
        Game_data.Obtained_constraints.append([None]*N)
        Game_data.Regrets.append(  [None]*N )            
        Game_data.Cum_constraints.append([0]*N)

        # Assign payoffs and compute regrets
        for i in range(N):
            idx_rev_contexts = contexts.index(rev_contexts[t])
            Game_data.Obtained_payoffs[t][i] = Assign_payoffs(Game_data.Played_actions[t], r[i], idx_rev_contexts, K, len(contexts) )
            Game_data.Obtained_constraints[t][i] = Assign_constraints(Game_data.Played_actions[t][i], g[i])
            Game_data.Cum_constraints[t+1][i] = np.array(Game_data.Cum_constraints[t][i] + max(0,Game_data.Obtained_constraints[t][i]))
            
            for a in range(K):
                modified_outcome = np.copy(Game_data.Played_actions[t])
                idx_context = contexts.index(rev_contexts[t])
                if (types[i] == "c_random") or (types[i] == "cGPMW") or (types[i] == "cAdaNormalGP") or (types[i] == "czAdaNormalGP"):
                    for z in range(len(contexts)):
                        if (Player[i].lcb_constraint_est[a] <= 0) and (rev_contexts[t] == contexts[z]):
                            modified_outcome[i] = a
                            Game_data.Cum_payoffs[i][z][a] = np.array(Game_data.Cum_payoffs[i][z][a] + Assign_payoffs( modified_outcome, r[i], idx_context, K, len(contexts) ))
                else:
                    for z in range(len(contexts)):
                        if (rev_contexts[t] == contexts[z]):
                            modified_outcome[i] = a
                            Game_data.Cum_payoffs[i][z][a] = np.array(Game_data.Cum_payoffs[i][z][a] + Assign_payoffs( modified_outcome, r[i], idx_context, K, len(contexts) ))
                
            if (types[i] == "zGPMW") or (types[i] == "czAdaNormalGP"):
                Game_data.Policies[i] = np.argmax(Game_data.Cum_payoffs[i], axis=1)
            else:
                Game_data.Policies[i] = np.array([np.argmax(np.sum([Game_data.Cum_payoffs[i][z] for z in range(Z)], axis =0))]*len(contexts))
            maxi = []
            for z in range(len(contexts)):
                maxi.append(np.max([Game_data.Cum_payoffs[i][z][a] for a in range(K) if (g[i][a] <= 0)]))
            Game_data.Regrets[t][i] = ( sum(maxi) -  sum([  Game_data.Obtained_payoffs[x][i] for x in range(t+1)]) ) / (t+1)
            
        # Update players mixed strategy
        Game_data.UCB_vectors_r.append( [None]*N)
        for i in range(N):

            if Player[i].type == "random":
                Player[i].Update()

            if Player[i].type == "c_random":
                last_own_action = [np.array(Game_data.Played_actions[t][i])]
                last_constraint = [Game_data.Obtained_constraints[t][i] + noises_g[t]]
                Player[i].GP_update_g(last_own_action,last_constraint, sigma1)
                Player[i].Update()
                
            if Player[i].type == "GPMW":
                last_actions = Game_data.Played_actions[t]
                last_payoff = [Game_data.Obtained_payoffs[t][i] + noises_r[t]]
                # Update GP posterior
                Player[i].GP_update( last_actions, last_payoff , sigma0)
                # Update Weights
                Player[i].Update()
                Game_data.UCB_vectors_r[t][i] = Player[i].ucb_reward_est.tolist()
         
            if Player[i].type == "zGPMW":
                last_actions = Game_data.Played_actions[t]
                last_payoff = [Game_data.Obtained_payoffs[t][i] + noises_r[t]]
                last_context = rev_contexts[t]
                # Update GP posterior
                Player[i].GP_update(last_actions, last_payoff, last_context, sigma0)
                # Update Weights
                Player[i].Update(last_context)
                Game_data.UCB_vectors_r[t][i] = Player[i].ucb_reward_est.tolist()
         
            if Player[i].type == "cGPMW":
                last_actions = Game_data.Played_actions[t]
                last_payoff = [Game_data.Obtained_payoffs[t][i] + noises_r[t] ]
                last_own_action = [np.array(Game_data.Played_actions[t][i])]
                last_constraint = [Game_data.Obtained_constraints[t][i] + noises_g[t]]
                # Update GP posteriors
                Player[i].GP_update_r(last_actions, last_payoff, sigma0)
                Player[i].GP_update_g(last_own_action,last_constraint, sigma1)
                # Update weights
                Player[i].Update()
                Game_data.UCB_vectors_r[t][i] = Player[i].ucb_reward_est.tolist()
         
            if Player[i].type == "cAdaNormalGP":
                last_actions = Game_data.Played_actions[t]
                last_payoff = [Game_data.Obtained_payoffs[t][i] + noises_r[t] ]
                last_own_action = [np.array(Game_data.Played_actions[t][i])]
                last_constraint = [Game_data.Obtained_constraints[t][i] + noises_g[t]]
                # Update GP posteriors
                Player[i].GP_update_r(last_actions, last_payoff, sigma0)
                Player[i].GP_update_g(last_own_action,last_constraint, sigma1)
                # Update weights
                Player[i].Update()
                Game_data.UCB_vectors_r[t][i] = Player[i].ucb_reward_est.tolist()
         
            if Player[i].type == "czAdaNormalGP":
                last_actions = Game_data.Played_actions[t]
                last_payoff = [Game_data.Obtained_payoffs[t][i] + noises_r[t] ]
                last_context = rev_contexts[t]
                last_own_action = [np.array(Game_data.Played_actions[t][i])]
                last_constraint = [np.array(Game_data.Obtained_constraints[t][i] + noises_g[t])]
                # Update GP posteriors
                Player[i].GP_update_r(last_actions, last_payoff, last_context, sigma0)
                Player[i].GP_update_g(last_own_action,last_constraint, sigma1)
                # Update weights
                Player[i].Update(last_context)
                Game_data.UCB_vectors_r[t][i] = Player[i].ucb_reward_est.tolist()
                

    return Game_data , Player                

#%% 
############################# Generate game data ##################################################
'''
Generate reward and constraint functions for each player
'''
seed = 9
np.random.seed(seed)
Runs = 10
N = 3
K = 7
Z = 5 # finite (small) context space: Z=1,...,|Z|

save_r = []
save_g = []
for run in range(Runs):
    for i in range(N):
        print('i:' + str(i))
        reward, input = Generate_r(N,K,Z)
        reward = reward # *0.1
        save_r.append(reward)
        constraint = Generate_g(K)
        constraint = constraint*10
        save_g.append(constraint)
    print('run:' + str(run))

# Save the generated reward and constrained functions
with open(f"reward+constraints_{Runs}_seed={seed}.pckl", 'wb') as file:
    pickle.dump(N, file)
    pickle.dump(K, file)
    pickle.dump(Z, file)
    pickle.dump(save_r , file)
    pickle.dump(save_g, file)


#%%
'''
Load saved reward and constrained functions
'''
seed = 9
Runs = 10
N = 3
K = 7
Z = 5
with open(f"reward+constraints_{Runs}_seed={seed}.pckl", 'rb') as file:
   
    N = pickle.load(file)
    K = pickle.load(file)
    Z = pickle.load(file)
    save_r = pickle.load(file)
    save_g = pickle.load(file)



#%% 
##################################### Simulate Game #############################################################
# Add later after class GameData is defined
np.random.seed(seed)
T = 1000
# Generate the set of revealed contexts
contexts = [(z+1) for z in range(Z)]
rev_contexts = np.random.choice(np.arange(1,Z+1),T) # set back to Z+1 instead of 3 

N_types = []
N_types.append(['random','random','random'])
#N_types.append(['c_random','random','random'])
#N_types.append(['cGPMW','random','random'])
N_types.append(['GPMW','random','random'])
N_types.append(['zGPMW','random','random'])
N_types.append(['cAdaNormalGP','random','random'])
N_types.append(['czAdaNormalGP','random','random'])

sigma0 = 0.1 # 0.5
sigma1 = 0.1
version = 1 # finite (small) number of contexts

# Run game for different player types
for type in range(len(N_types)):

    print('Player_type: ' + str(N_types[type]))

    UCB_r = []
    UCB_g = []
    LCB_g = []
    played_actions = []
    obtained_payoffs = []
    cum_payoffs = []
    cum_constraints = []
    regrets = []
    weights = []
    for run in range(Runs):

        Games_data, Player = RunGame(N,K,T,save_r[(run*N):(run*N+3)],save_g[(run*N):(run*N+3)],N_types[type],sigma0,sigma1,contexts,rev_contexts, version)
        UCB_r.append(Games_data.UCB_vectors_r)
        UCB_g.append(Games_data.UCB_vectors_g)
        LCB_g.append(Games_data.LCB_vectors_g)
        played_actions.append(Games_data.Played_actions)
        obtained_payoffs.append(Games_data.Obtained_payoffs)
        cum_payoffs.append(Games_data.Cum_payoffs)
        cum_constraints.append(Games_data.Cum_constraints)
        regrets.append(Games_data.Regrets)
        for i in range(N):
            weights.append(Player[i].weights)

        print('Run: ' + str(run))
    
    with open(f'Player_{N_types[type][0]}_T={T}_.pckl', 'wb') as file:
        pickle.dump(N_types[type] , file)
        pickle.dump(save_r, file)
        pickle.dump(save_g, file)
        pickle.dump(Runs, file)
        pickle.dump(N, file)
        pickle.dump(T, file)
        pickle.dump(K, file)
        pickle.dump(contexts, file)
        pickle.dump(rev_contexts, file)
        pickle.dump(version, file)
        pickle.dump(UCB_r, file)
        pickle.dump(UCB_g, file)
        pickle.dump(LCB_g, file)
        pickle.dump(played_actions, file)
        pickle.dump(obtained_payoffs, file)
        pickle.dump(cum_payoffs, file)
        pickle.dump(cum_constraints, file)
        pickle.dump(regrets, file)
        pickle.dump(weights, file)
    

    

