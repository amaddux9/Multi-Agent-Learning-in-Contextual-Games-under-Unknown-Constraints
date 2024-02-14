import numpy as np
import GPy
import itertools

" Type of Players "

class Player_random:
    def __init__(self,K,i):
        self.type = "random"
        self.K = K
        self.idx_player = i
        self.weights = np.ones(K)

    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)
    
    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))
    
    def Update(self):
        self.weights = np.ones(self.K)
    


class Player_c_random:
    def __init__(self,K,k_g):
        self.type = "c_random"
        self.K = K
        #self.idx_player = i
        self.weights = np.ones(K)
        self.ucb_constraint_est = np.zeros(K)
        self.lcb_constraint_est = np.zeros(K)
        self.mean_constraint_est = 0*np.ones(K)
        self.std_constraint_est = np.zeros(K)
        self.kernel_g = k_g
        self.history_own_actions = []
        self.history_constraints = []
        self.input = np.array([[a] for a in range(K)])  


        # Compute GP prior on constraint
        """
        Initialize the variance matrix (K(a_i,a_i) for all i) for the constraint function
        """
        beta_t = 2.0
        var_constraint_est = np.diag(self.kernel_g.K(self.input,self.input))
        self.ucb_constraint_est =  self.mean_constraint_est + beta_t*np.sqrt(var_constraint_est)
        self.lcb_constraint_est =  self.mean_constraint_est - beta_t*np.sqrt(var_constraint_est)

        self.std_constraint_est = np.sqrt(var_constraint_est)
   
        
    def mixed_strategy(self):
        indic = [1 if ai<=0 else 0 for ai in self.lcb_constraint_est]
        return np.multiply(indic,self.weights) / np.sum(np.multiply(indic,self.weights))
    
    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))
    
    def Update(self):
        self.weights = np.ones(self.K)

    def GP_update_g(self, last_action, last_constraint, sigma_noise_g):
        self.history_own_actions.append(last_action)
        self.history_constraints.append(last_constraint)
        beta_t = 2.0

        X = np.array(self.history_own_actions)
        Y = np.array(self.history_constraints)
        model = GPy.models.GPRegression(X,Y, self.kernel_g)
        model.Gaussian_noise.fix(sigma_noise_g**2)

        mu, var = model.predict(self.input)
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))
    
        self.ucb_constraint_est = mu + beta_t*sigma
        self.lcb_constraint_est = mu - beta_t*sigma
        self.mean_constraint_est = mu
        self.std_constraint_est = sigma


class Player_GPMW:
    def __init__(self, K, T, i, min_payoff, payoff_range, kernel_r):
        self.type = "GPMW"
        self.idx_player = i 
        self.K = K
        self.min_payoff = min_payoff
        self.payoff_range = payoff_range
        self.weights = np.ones(K)
        self.T = T 
        
        self.cum_losses= np.zeros(K) 
        self.mean_reward_est = 0*np.ones(K)
        self.std_reward_est = np.zeros(K)
        self.ucb_reward_est = np.zeros(K)
        self.gamma_t = np.sqrt(8*np.log(K) / T)
        self.kernel = kernel_r
        
        self.history_actions = []
        self.history_payoffs = []


    def mixed_strategy(self):
        return self.weights / np.sum(self.weights)

    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))

    def GP_update(self, last_actions, last_payoff, sigma_noise_r):

        self.history_actions.append(last_actions)
        self.history_payoffs.append(last_payoff)

        beta_t = 2 

        X = np.array(self.history_actions)
        Y = np.array(self.history_payoffs)
        model = GPy.models.GPRegression(X, Y, self.kernel)
        model.Gaussian_noise.fix(sigma_noise_r**2)

        modified_outcome = []
        for ai in range(self.K):
            help = np.copy(last_actions)
            help[self.idx_player] = ai
            modified_outcome.append(help)
        mu, var = model.predict(np.array(modified_outcome))
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))

        self.ucb_reward_est = mu + beta_t * sigma
        self.mean_reward_est = mu
        self.std_reward_est = sigma

    def Update(self):
        payoffs = np.array(self.ucb_reward_est)
        payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
        payoffs = np.minimum(payoffs, (self.min_payoff + self.payoff_range)* np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.payoff_range))
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        self.cum_losses = self.cum_losses + losses

        gamma_t = self.gamma_t
        self.weights = np.exp(np.multiply(gamma_t, -self.cum_losses))



class Player_zGPMW:
    def __init__(self, K, T, i, min_payoff, payoff_range, k1, k2, version, first_context):
        self.type = "zGPMW"
        self.K = K
        self.idx_player = i 
        self.min_payoff = min_payoff
        self.payoff_range = payoff_range
        self.weights = []
        self.weights.append(np.ones(K))
        self.T = T

        self.gamma_t = np.sqrt(8*np.log(K) / T)
        self.kernel = k1 * k2

        self.history_payoffs = [[]]
        self.history_actions = [[]]
        self.history_inputs = [[]] # action profiles and contexts combined
        
        self.cum_losses=[]
        self.cum_losses.append(np.zeros(K))
        self.mean_reward_est = 0*np.ones(K)
        self.std_reward_est = np.zeros(K)
        self.ucb_reward_est = np.zeros(K)

        self.version  = version
        self.set_history_contexts = []
        self.set_history_contexts.append(first_context)

    def mixed_strategy(self, c_idx_t):
        return self.weights[c_idx_t] / np.sum(self.weights[c_idx_t])

    def sample_action(self,last_context):
        distances = np.array([np.linalg.norm([last_context - self.set_history_contexts[c]], 1) for c in range(len(self.set_history_contexts))])
        c_idx_t = distances.argmin()
        return np.random.choice(self.K, p=np.array(self.mixed_strategy(c_idx_t)))

    def Update(self, last_context):
        distances = np.array([np.linalg.norm([last_context - self.set_history_contexts[c]], 1) for c in range(len(self.set_history_contexts))])
        c_idx_t = distances.argmin()
        payoffs = np.array(self.ucb_reward_est)
        payoffs = np.maximum(payoffs, self.min_payoff * np.ones(self.K))
        payoffs = np.minimum(payoffs, (self.min_payoff + self.payoff_range)* np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff) / (self.payoff_range))
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        self.cum_losses[c_idx_t] = self.cum_losses[c_idx_t] + losses

        gamma_t = self.gamma_t
        self.weights[c_idx_t] = np.exp(np.multiply(gamma_t, -self.cum_losses[c_idx_t]))


    def GP_update(self, last_actions, last_payoff, last_context, sigma_noise_r):

        distances = np.array([np.linalg.norm([last_context - self.set_history_contexts[c]], 1) for c in range(len(self.set_history_contexts))])
        if self.version == 2:
            if distances.min() < epsilon:
                c_idx_t = distances.argmin()
            else:
                self.set_history_contexts.append(last_context)
                self.weights.append(np.ones(self.K))
                self.history_payoffs.append([])
                self.history_actions.append([])
                self.history_inputs.append([])
                self.cum_losses.append(np.zeros(self.K))
                c_idx_t = self.set_history_contexts.index(last_context)
        elif self.version == 1:
            if self.set_history_contexts.count(last_context) > 0:
                c_idx_t = distances.argmin()
            else:
                self.set_history_contexts.append(last_context)
                self.weights.append(np.ones(self.K))
                self.history_payoffs.append([])
                self.history_actions.append([])
                self.history_inputs.append([])
                self.cum_losses.append(np.zeros(self.K))
                c_idx_t = self.set_history_contexts.index(last_context)
        
        self.history_actions[c_idx_t].append(last_actions)
        self.history_inputs[c_idx_t].append(np.concatenate((last_actions,last_context), axis = None))
        self.history_payoffs[c_idx_t].append(last_payoff)

        beta_t = 2 

        X = np.array(self.history_inputs[c_idx_t])
        Y = np.array(self.history_payoffs[c_idx_t])
        model = GPy.models.GPRegression(X,Y, self.kernel)
        model.Gaussian_noise.fix(sigma_noise_r**2)

        modified_outcome = []
        for ai in range(self.K):
            help1 = np.copy(last_actions)
            help2 = np.copy(last_context)
            help = np.concatenate((help1,help2), axis = None)
            help[self.idx_player] = ai
            modified_outcome.append(help)

        mu, var = model.predict(np.array(modified_outcome))
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))

        self.ucb_reward_est = mu + beta_t * sigma
        self.mean_reward_est = mu
        self.std_reward_est = sigma


class Player_cGPMW: #constrained GPMW
    def __init__(self, K,T,i, min_payoff, payoffs_range, kernel_r, kernel_g):
        self.type = "cGPMW"
        self.idx_player = i
        self.K = K
        self.min_payoff = min_payoff
        self.payoffs_range = payoffs_range
        self.weights = np.ones(K)
        self.ucb_reward_est = np.zeros(K)
        self.ucb_constraint_est = np.zeros(K)
        self.lcb_constraint_est = np.zeros(K)
        self.cum_losses = np.zeros(K)
        self.gamma_t = np.sqrt(8*np.log(K) / T)
        self.mean_reward_est = 0*np.ones(K) 
        self.std_reward_est = np.zeros(K)
        self.mean_constraint_est = 0*np.ones(K)
        self.std_constraint_est = np.zeros(K)
        self.kernel_r = kernel_r
        self.kernel_g = kernel_g
        self.history_actions = []
        self.history_payoffs = []
        self.history_own_actions = []
        self.history_constraints = []
        self.input = np.array([[a] for a in range(K)])  


        beta_t = 2.0
        var_constraint_est = np.diag(self.kernel_g.K(self.input,self.input))
        self.ucb_constraint_est =  self.mean_constraint_est + beta_t*np.sqrt(var_constraint_est)
        self.lcb_constraint_est =  self.mean_constraint_est - beta_t*np.sqrt(var_constraint_est)

        self.std_constraint_est = np.sqrt(var_constraint_est)
   
    def mixed_strategy(self):
        indic = [1 if ai<=0 else 0 for ai in self.lcb_constraint_est]
        return np.multiply(indic,self.weights) / np.sum(np.multiply(indic,self.weights))
    
    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))
        
    
    def GP_update_r(self, last_actions, last_payoff, sigma_noise_r):
        
        self.history_actions.append(last_actions)
        self.history_payoffs.append(last_payoff)

        beta_t = 2 

        X = np.array(self.history_actions)
        Y = np.array(self.history_payoffs)
        model = GPy.models.GPRegression(X,Y, self.kernel_r)
        model.Gaussian_noise.fix(sigma_noise_r**2)

        modified_outcome = []
        for ai in range(self.K):
            help = np.copy(last_actions)
            help[self.idx_player] = ai
            modified_outcome.append(help)
        mu, var = model.predict(np.array(modified_outcome))
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))

        self.ucb_reward_est = mu + beta_t * sigma
        self.mean_reward_est = mu
        self.std_reward_est = sigma

    def GP_update_g(self, last_action, last_constraint, sigma_noise_g):
        
        self.history_own_actions.append(last_action)
        self.history_constraints.append(last_constraint)
        beta_t = 2.0

        X = np.array(self.history_own_actions)
        Y = np.array(self.history_constraints)
        model = GPy.models.GPRegression(X,Y, self.kernel_g)
        model.Gaussian_noise.fix(sigma_noise_g**2)

        mu, var = model.predict(self.input)
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))
    
        self.ucb_constraint_est = mu + beta_t*sigma
        self.lcb_constraint_est = mu - beta_t*sigma
        self.mean_constraint_est = mu
        self.std_constraint_est = sigma


    def Update(self):
        payoffs = np.array(self.ucb_reward_est)
        payoffs = np.maximum(payoffs, self.min_payoff*np.ones(self.K))
        payoffs = np.minimum(payoffs, (self.min_payoff + self.payoffs_range)*np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff)/self.payoffs_range)
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        self.cum_losses = self.cum_losses + losses

        gamma_t = self.gamma_t
        self.weights = np.exp(np.multiply(gamma_t, -self.cum_losses))

        

class Player_cAdaNormalGP: 
    def __init__(self, K,i, min_payoff, payoffs_range, kernel_r, kernel_g):
        self.type = "cAdaNormalGP"
        self.idx_player = i
        self.K = K
        self.min_payoff = min_payoff
        self.payoffs_range = payoffs_range
        self.weights = np.ones(K)
        self.ucb_reward_est = np.zeros(K)
        self.ucb_constraint_est = np.zeros(K)
        self.lcb_constraint_est = np.zeros(K)
        self.cum_ucb_regret = np.zeros(K)
        self.cum_abs_ucb_regret = np.zeros(K)
        self.mean_reward_est = 0*np.ones(K) 
        self.std_reward_est = np.zeros(K)
        self.mean_constraint_est = 0*np.ones(K)
        self.std_constraint_est = np.zeros(K)
        self.kernel_r = kernel_r
        self.kernel_g = kernel_g
        self.history_actions = []
        self.history_payoffs = []
        self.history_own_actions = []
        self.history_constraints = []
        self.input = np.array([[a] for a in range(K)])  


        beta_t = 2.0
        var_constraint_est = np.diag(self.kernel_g.K(self.input,self.input))
        self.ucb_constraint_est =  self.mean_constraint_est + beta_t*np.sqrt(var_constraint_est)
        self.lcb_constraint_est =  self.mean_constraint_est - beta_t*np.sqrt(var_constraint_est)
        self.std_constraint_est = np.sqrt(var_constraint_est)
   
    def mixed_strategy(self):
        indic = [1 if ai<=0 else 0 for ai in self.lcb_constraint_est]
        return np.multiply(indic,self.weights) / np.sum(np.multiply(indic,self.weights))
    
    def sample_action(self):
        return np.random.choice(self.K, p=np.array(self.mixed_strategy()))
        
    
    def GP_update_r(self, last_actions, last_payoff, sigma_noise_r):
        
        self.history_actions.append(last_actions)
        self.history_payoffs.append(last_payoff)

        beta_t = 2 

        X = np.array(self.history_actions)
        Y = np.array(self.history_payoffs)
        model = GPy.models.GPRegression(X,Y, self.kernel_r)
        model.Gaussian_noise.fix(sigma_noise_r**2)

        modified_outcome = []
        for ai in range(self.K):
            help = np.copy(last_actions)
            help[self.idx_player] = ai
            modified_outcome.append(help)
        mu, var = model.predict(np.array(modified_outcome))
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))

        self.ucb_reward_est = mu + beta_t * sigma
        self.mean_reward_est = mu
        self.std_reward_est = sigma

    def GP_update_g(self, last_action, last_constraint, sigma_noise_g):
        
        self.history_own_actions.append(last_action)
        self.history_constraints.append(last_constraint)
        beta_t = 2.0

        X = np.array(self.history_own_actions)
        Y = np.array(self.history_constraints)
        model = GPy.models.GPRegression(X,Y, self.kernel_g)
        model.Gaussian_noise.fix(sigma_noise_g**2)

        mu, var = model.predict(self.input)
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))
    
        self.ucb_constraint_est = mu + beta_t*sigma
        self.lcb_constraint_est = mu - beta_t*sigma
        self.mean_constraint_est = mu
        self.std_constraint_est = sigma

            
    def Update(self):
        payoffs = np.array(self.ucb_reward_est)
        payoffs = np.maximum(payoffs, self.min_payoff*np.ones(self.K))
        payoffs = np.minimum(payoffs, (self.min_payoff + self.payoffs_range)*np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff)/self.payoffs_range)
        
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        cum_ucb_regret_prev = np.array(self.cum_ucb_regret)
        cum_abs_ucb_regret_prev = np.array(self.cum_abs_ucb_regret)
        cum_ucb_regret = np.zeros(self.K)
        cum_abs_ucb_regret = np.zeros(self.K)
        ai_t = self.history_actions[-1][self.idx_player]
        indic = [1 if ai<=0 else 0 for ai in self.lcb_constraint_est]
        cum_ucb_regret = cum_ucb_regret_prev + np.multiply(indic,losses[ai_t]*np.ones(self.K)-losses)
        cum_abs_ucb_regret = cum_abs_ucb_regret_prev + np.abs(np.multiply(indic,losses[ai_t]*np.ones(self.K)-losses))
        
        self.cum_ucb_regret = cum_ucb_regret
        self.cum_abs_ucb_regret = cum_abs_ucb_regret         
        
        self.weights = 1/2*(np.exp(np.divide(np.square(np.maximum(np.zeros(self.K),cum_ucb_regret+np.ones(self.K))),3*(cum_abs_ucb_regret+np.ones(self.K))))-np.exp(np.divide(np.square(np.maximum(np.zeros(self.K),cum_ucb_regret-np.ones(self.K))),3*(cum_abs_ucb_regret+np.ones(self.K)))))            
        


class Player_czAdaNormalGP: 
    def __init__(self, K,i, min_payoff, payoffs_range, k1, k2, kernel_g, version,first_context):
        self.type = "czAdaNormalGP"
        self.idx_player = i
        self.K = K
        self.min_payoff = min_payoff
        self.payoffs_range = payoffs_range
        self.weights = []
        self.weights.append(np.ones(K))
        
        self.ucb_reward_est = np.zeros(K)
        self.cum_ucb_regret = []
        self.cum_ucb_regret.append(np.zeros(K))
        self.cum_abs_ucb_regret = []
        self.cum_abs_ucb_regret.append(np.zeros(K))
        self.mean_reward_est = 0*np.ones(K) 
        self.std_reward_est = np.zeros(K)

        self.ucb_constraint_est = np.zeros(K)
        self.lcb_constraint_est = np.zeros(K)
        self.mean_constraint_est = 0*np.ones(K)
        self.std_constraint_est = np.zeros(K)

        self.kernel_r = k1*k2
        self.kernel_g = kernel_g

        self.history_payoffs = [[]]
        self.history_actions = [[]]
        self.history_inputs = [[]] # action profiles and contexts combined
        self.history_own_actions = []
        self.history_constraints = []
        self.version  = version
        self.set_history_contexts = []
        self.set_history_contexts.append(first_context)
        self.input = np.array([[a] for a in range(K)])  


        beta_t = 2.0
        var_constraint_est = np.diag(self.kernel_g.K(self.input,self.input))
        self.ucb_constraint_est =  self.mean_constraint_est + beta_t*np.sqrt(var_constraint_est)
        self.lcb_constraint_est =  self.mean_constraint_est - beta_t*np.sqrt(var_constraint_est)

        self.std_constraint_est = np.sqrt(var_constraint_est)
   
    def mixed_strategy(self, c_idx_t):
        indic = [1 if ai<=0 else 0 for ai in self.lcb_constraint_est]
        return np.multiply(indic,self.weights[c_idx_t]) / np.sum(np.multiply(indic,self.weights[c_idx_t]))
    
    def sample_action(self, last_context):
        distances = np.array([np.linalg.norm([last_context - self.set_history_contexts[c]], 1) for c in range(len(self.set_history_contexts))])
        c_idx_t = distances.argmin()
        return np.random.choice(self.K, p=np.array(self.mixed_strategy(c_idx_t)))
        
    def GP_update_r(self, last_actions, last_payoff, last_context, sigma_noise_r):
    
        distances = np.array([np.linalg.norm([last_context - self.set_history_contexts[c]], 1) for c in range(len(self.set_history_contexts))])
        if self.version == 2:
            epsilon  = 0.1 
            if distances.min() < epsilon:
                c_idx_t = distances.argmin()
            else:
                self.set_history_contexts.append(last_context)
                self.weights.append(np.ones(self.K))
                self.history_payoffs.append([])
                self.history_actions.append([])
                self.history_inputs.append([])
                self.cum_ucb_regret.append(np.zeros(self.K))
                self.cum_abs_ucb_regret.append(np.zeros(self.K))
                c_idx_t = self.set_history_contexts.index(last_context)
        elif self.version == 1:
            if self.set_history_contexts.count(last_context) > 0:
                c_idx_t = distances.argmin()
            else:
                self.set_history_contexts.append(last_context)
                self.weights.append(np.ones(self.K))
                self.history_payoffs.append([])
                self.history_actions.append([])
                self.history_inputs.append([])
                self.cum_ucb_regret.append(np.zeros(self.K))
                self.cum_abs_ucb_regret.append(np.zeros(self.K))
                c_idx_t = self.set_history_contexts.index(last_context)
        
        self.history_actions[c_idx_t].append(last_actions)
        self.history_inputs[c_idx_t].append(np.concatenate((last_actions,last_context), axis = None))
        self.history_payoffs[c_idx_t].append(last_payoff)

        beta_t = 2 

        X = np.array(self.history_inputs[c_idx_t])
        Y = np.array(self.history_payoffs[c_idx_t])
        model = GPy.models.GPRegression(X,Y, self.kernel_r)
        model.Gaussian_noise.fix(sigma_noise_r**2)

        modified_outcome = []
        for ai in range(self.K):
            help1 = np.copy(last_actions)
            help2 = np.copy(last_context)
            help = np.concatenate((help1,help2), axis = None)
            help[self.idx_player] = ai
            modified_outcome.append(help)

        mu, var = model.predict(np.array(modified_outcome))
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))

        self.ucb_reward_est = mu + beta_t * sigma
        self.mean_reward_est = mu
        self.std_reward_est = sigma


    def GP_update_g(self, last_action, last_constraint, sigma_noise_g):
        
        self.history_own_actions.append(last_action)
        self.history_constraints.append(last_constraint)
        beta_t = 2.0

        X = np.array(self.history_own_actions)
        Y = np.array(self.history_constraints)
        model = GPy.models.GPRegression(X,Y, self.kernel_g)
        model.Gaussian_noise.fix(sigma_noise_g**2)

        mu, var = model.predict(self.input)
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))
    
        self.ucb_constraint_est = mu + beta_t*sigma
        self.lcb_constraint_est = mu - beta_t*sigma
        self.mean_constraint_est = mu
        self.std_constraint_est = sigma

            
    def Update(self, last_context):
        distances = np.array([np.linalg.norm([last_context - self.set_history_contexts[c]], 1) for c in range(len(self.set_history_contexts))])
        c_idx_t = distances.argmin()
        payoffs = np.array(self.ucb_reward_est)
        payoffs = np.maximum(payoffs, self.min_payoff*np.ones(self.K))
        payoffs = np.minimum(payoffs, (self.min_payoff + self.payoffs_range)*np.ones(self.K))
        payoffs_scaled = np.array((payoffs - self.min_payoff)/self.payoffs_range)
        
        losses = np.ones(self.K) - np.array(payoffs_scaled)
        cum_ucb_regret_prev = np.array(self.cum_ucb_regret[c_idx_t])
        cum_abs_ucb_regret_prev = np.array(self.cum_abs_ucb_regret[c_idx_t])
        cum_ucb_regret = np.zeros(self.K)
        cum_abs_ucb_regret = np.zeros(self.K)
        ai_t = self.history_actions[c_idx_t][-1][self.idx_player]
        indic = [1 if ai<=0 else 0 for ai in self.lcb_constraint_est]
        cum_ucb_regret = cum_ucb_regret_prev + np.multiply(indic,losses[ai_t]*np.ones(self.K)-losses)
        cum_abs_ucb_regret = cum_abs_ucb_regret_prev + np.abs(np.multiply(indic,losses[ai_t]*np.ones(self.K)-losses))
        
        self.cum_ucb_regret[c_idx_t] = cum_ucb_regret
        self.cum_abs_ucb_regret[c_idx_t] = cum_abs_ucb_regret         
        
        self.weights[c_idx_t] = 1/2*(np.exp(np.divide(np.square(np.maximum(np.zeros(self.K),self.cum_ucb_regret[c_idx_t]+np.ones(self.K))),3*(self.cum_abs_ucb_regret[c_idx_t]+np.ones(self.K))))-np.exp(np.divide(np.square(np.maximum(np.zeros(self.K),self.cum_ucb_regret[c_idx_t]-np.ones(self.K))),3*(self.cum_abs_ucb_regret[c_idx_t]+np.ones(self.K)))))            
        

def Assign_payoffs(outcome , r, idx_context,K,Z):
    # outcome = (a1,...,aN)
    # r - reward function
    return r[K**2*Z*outcome[0]+K*Z*outcome[1]+Z*outcome[2]+idx_context]

def Assign_constraints(outcome, constraint_vector):
    g = constraint_vector
    return g[outcome]
