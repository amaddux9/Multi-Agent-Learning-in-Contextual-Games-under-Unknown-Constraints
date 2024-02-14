import numpy as np
import GPy
import itertools

class Player_cAdaNormalGP: 
    def __init__(self, params_args, i, min_loss, loss_range, kernel_loss, kernel_g, dev_threshold):
        
        P, SP_day, shift_night = params_args
        self.P = P
        self.SP_day = SP_day
        self.shift_night = shift_night
        self.len_P = len(P)
        self.len_SP_day = len(SP_day)
        self.len_shift_night = len(shift_night)
        self.type = "cAdaNormalGP"
        self.idx_player = i
        self.min_loss = min_loss
        self.loss_range = loss_range
        self.dev_threshold = dev_threshold
        self.weights = np.ones(self.len_P * self.len_SP_day * self.len_shift_night)
        self.lcb_loss_est = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.cum_lcb_regret = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.cum_abs_lcb_regret = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.mean_loss_est = 0*np.ones(self.len_P * self.len_SP_day * self.len_shift_night)
        self.std_loss_est = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.ucb_constraint_est = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.lcb_constraint_est = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.mean_constraint_est = 0*np.ones(self.len_P * self.len_SP_day * self.len_shift_night)
        self.std_constraint_est = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.kernel_loss = kernel_loss
        self.kernel_g = kernel_g
        self.history_actions = []
        self.history_losses = []
        self.history_own_actions = []
        self.history_constraints = []
        self.idx_list = np.array(list(itertools.product(range(self.len_P),range(self.len_SP_day),range(self.len_shift_night))))
        self.params_list = np.array(list(itertools.product(P, SP_day, shift_night)))
    
        
        # Compute GP prior on constraint
        """
        Initialize the kernel matrix (K(a_i,a_j) for all i,j) and 
        the variance matrix (K(a_i,a_i) for all i) for the 
        constraint function
        """
        var_constraint_est = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        beta_t = 3.0
        var_constraint_est = np.diag(self.kernel_g.K(self.params_list, self.params_list))
        self.lcb_constraint_est =  self.mean_constraint_est - beta_t*np.sqrt(var_constraint_est)
        self.std_constraint_est = np.sqrt(var_constraint_est)


    def mixed_strategy(self):
        indic = [1 if ai<= self.dev_threshold else 0 for ai in self.lcb_constraint_est]
        if np.sum(np.multiply(indic,self.weights)) == 0:
            self.weights = np.ones(len(self.params_list))
        current_weights = np.multiply(indic,self.weights) / np.sum(np.multiply(indic,self.weights))
        return current_weights
    
    def sample_action(self):
        idx = np.random.choice(self.len_P * self.len_SP_day * self.len_shift_night, p=np.array(self.mixed_strategy()))
        param_idx = self.idx_list[idx]
        return [self.P[param_idx[0]], self.SP_day[param_idx[1]], self.shift_night[param_idx[2]]]

    
    def GP_update_l(self, last_energy, last_loss, true_energy_hourly, sim_days, sigma_noise_l):
        
        last_energy_concatenate = np.concatenate((last_energy[0], last_energy[1], last_energy[2]),axis = None)
        self.history_actions.append(last_energy_concatenate)
        self.history_losses.append(last_loss)

        beta_t = 2.0 

        X = np.array(self.history_actions)
        Y = np.array(self.history_losses)
        model = GPy.models.GPRegression(X,Y, self.kernel_loss)
        model.Gaussian_noise.fix(sigma_noise_l**2)

        modified_outcome = []
        for idx in range(len(self.params_list)): 
            helper = np.copy(last_energy_concatenate)
            if self.idx_player == 0:
                helper[0 : (sim_days*24)] = true_energy_hourly[idx]
            elif self.idx_player == 1:
                helper[1*(sim_days*24) : 2*(sim_days*24)] = true_energy_hourly[idx]
            else:
                helper[2*(sim_days*24) : 3*(sim_days*24)] = true_energy_hourly[idx]
            modified_outcome.append(helper)
        mu, var = model.predict(np.array(modified_outcome))
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))

        self.lcb_loss_est = mu - beta_t * sigma
        self.mean_loss_est = mu
        self.std_loss_est = sigma
        

    def GP_update_g(self, last_action, last_constraint, sigma_noise_g):
        
        self.history_own_actions.append(last_action)
        self.history_constraints.append(last_constraint)
        beta_t = 3.0

        X = np.array(self.history_own_actions)
        Y = np.array(self.history_constraints)
        model = GPy.models.GPRegression(X,Y, self.kernel_g)
        model.Gaussian_noise.fix(sigma_noise_g**2)

        mu, var = model.predict(self.params_list) 
        mu = mu.flatten()
        var = var.flatten()  
        sigma = np.sqrt(np.maximum(var, 1e-6))
        
        self.lcb_constraint_est = mu - beta_t*sigma
        self.mean_constraint_est = mu
        self.std_constraint_est = sigma
        

    def Update(self):
        losses = np.array(self.lcb_loss_est)
        losses = np.maximum(losses, self.min_loss*np.ones(len(self.params_list)))
        losses = np.minimum(losses, (self.min_loss + self.loss_range)*np.ones(len(self.params_list)))
        losses = np.array((losses - self.min_loss)/self.loss_range)

        cum_lcb_regret_prev = np.array(self.cum_lcb_regret)
        cum_abs_lcb_regret_prev = np.array(self.cum_abs_lcb_regret)
        
        indic = [1 if ai<= self.dev_threshold else 0 for ai in self.lcb_constraint_est]
        if np.sum(np.multiply(indic,self.weights)) == 0:
            self.weights = np.ones(len(self.params_list))
        normalized_weights = np.multiply(indic,self.weights) / np.sum(np.multiply(indic,self.weights))
        instant_regret = np.dot(normalized_weights,losses)*np.ones(len(losses)) - losses
        sleeping_instant_regret = np.multiply(indic, instant_regret)
        self.cum_lcb_regret = cum_lcb_regret_prev + sleeping_instant_regret
        self.cum_abs_lcb_regret = cum_abs_lcb_regret_prev + np.abs(sleeping_instant_regret)
        
        max1 = np.maximum( np.zeros(len(self.params_list)),
                           self.cum_lcb_regret+np.ones(len(self.params_list)))
                            
        expr1 = np.divide( np.square( max1 ),
                           3*(self.cum_abs_lcb_regret 
                            + np.ones(len(self.params_list)))
                        )
        
        max2 = np.maximum( np.zeros(len(self.params_list)),
                           self.cum_lcb_regret-np.ones(len(self.params_list)))
                                    
        expr2 = np.divide( np.square( max2 ),
                           3*(self.cum_abs_lcb_regret
                            + np.ones(len(self.params_list)))
                        )

        new_weights = 1/2*( np.exp(expr1) - np.exp(expr2) )   
        self.weights = new_weights/ np.sum(new_weights)
        

class Player_GPMW: 
    def __init__(self, T, params_args, i, min_loss, loss_range, kernel_loss):
        
        P, SP_day, shift_night = params_args
        self.P = P
        self.SP_day = SP_day
        self.shift_night = shift_night
        self.len_P = len(P)
        self.len_SP_day = len(SP_day)
        self.len_shift_night = len(shift_night)
        self.type = "GPMW"
        self.T = T
        self.idx_player = i
        self.min_loss = min_loss
        self.loss_range = loss_range
        self.weights = np.ones(self.len_P * self.len_SP_day * self.len_shift_night)
        self.lcb_loss_est = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.mean_loss_est = 0*np.ones(self.len_P * self.len_SP_day * self.len_shift_night) # why not np.zeros?
        self.std_loss_est = np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.gamma_t = np.sqrt(8*np.log(self.len_P * self.len_SP_day * self.len_shift_night) / T)
        self.cum_losses= np.zeros(self.len_P * self.len_SP_day * self.len_shift_night)
        self.kernel_loss = kernel_loss
        self.history_actions = []
        self.history_losses = []
        self.idx_list = np.array(list(itertools.product(range(self.len_P),range(self.len_SP_day),range(self.len_shift_night))))
        self.params_list = np.array(list(itertools.product(P, SP_day, shift_night)))
    
    
    def mixed_strategy(self):
        if np.sum(self.weights) == 0:
            self.weights = np.ones(len(self.params_list))
        return self.weights / np.sum(self.weights)
    

    def sample_action(self):
        idx = np.random.choice(self.len_P * self.len_SP_day * self.len_shift_night, p=np.array(self.mixed_strategy()))
        param_idx = self.idx_list[idx]
        return [self.P[param_idx[0]], self.SP_day[param_idx[1]], self.shift_night[param_idx[2]]]

    
    def GP_update_l(self, last_energy, last_loss, true_energy_hourly, sim_days, sigma_noise_l):
        
        last_energy_concatenate = np.concatenate((last_energy[0], last_energy[1], last_energy[2]), axis = None)
        self.history_actions.append(last_energy_concatenate)
        self.history_losses.append(last_loss)

        beta_t = 2 

        X = np.array(self.history_actions)
        Y = np.array(self.history_losses)
        model = GPy.models.GPRegression(X,Y, self.kernel_loss)
        model.Gaussian_noise.fix(sigma_noise_l**2)

        modified_outcome = []
        for idx in range(len(self.params_list)): 
            helper = np.copy(last_energy_concatenate)
            if self.idx_player == 0:
                helper[0 : (sim_days*24)] = true_energy_hourly[idx]
            elif self.idx_player == 1:
                helper[1*(sim_days*24) : 2*(sim_days*24)] = true_energy_hourly[idx]
            else:
                helper[2*(sim_days*24) : 3*(sim_days*24)] = true_energy_hourly[idx]
            modified_outcome.append(helper)
        mu, var = model.predict(np.array(modified_outcome))
        mu = mu.flatten()
        var = var.flatten()
        sigma = np.sqrt(np.maximum(var, 1e-6))

        self.lcb_loss_est = mu - beta_t * sigma
        self.mean_loss_est = mu
        self.std_loss_est = sigma

        
    def Update(self):
        losses = np.array(self.lcb_loss_est)
        losses = np.maximum(losses, self.min_loss*np.ones(len(self.params_list)))
        losses = np.minimum(losses, (self.min_loss + self.loss_range)*np.ones(len(self.params_list)))
        losses = np.array((losses - self.min_loss)/self.loss_range)
        self.cum_losses = self.cum_losses + losses

        gamma_t = self.gamma_t
        self.weights = np.exp(np.multiply(gamma_t, -self.cum_losses))
