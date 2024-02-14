#%%
import energym
import pandas as pd
import numpy as np
Celcius_to_Kelvin = 273.15


class PController(object):
    """
    Rule-based controller for heat pump control.

    Attributes
    ----------
    controls : list of str
        List of control inputs.
    observations : list of str
        List of zone temperature observations
    tol1 : float
        First threshold for deviation from the goal temperature.
    tol2 : float
        Second threshold for deviation from the goal temperature.
    nighttime_setback : bool
        Whether to use a nighttime setback.
    nighttime_start : int
        Hour to start the nighttime setback.
    nighttime_end : int
        Hour to end the nighttime setback.
    nighttime_temp : float
        Goal temperature during nighttime setback

    Methods
    -------
    get_control(obs, temp_sp, hour)
        Computes the control actions.
    """

    def __init__(
        self,
        control_list,
        P,
        SP_temp_night,
        SP_temp_day,
        nighttime_start=17,
        nighttime_end=6,
    ):
        """
        Parameters
        ----------
        control_list : list of str
            List containing all inputs
        P : float
            Gain for the P-controller.
        SP_temp_night : float
            Nighttime temperature set point.
        SP_temp_day : float
            Daytime temperature set point.
        nighttime_start : int, optional
            Hour to start the nighttime setback, by default 17
        nighttime_end : int, optional
            Hour to end the nighttime setback, by default 6
        
        Raises
        ------
        TypeError
            If wrong input types are detected.
        """
        self.controls = control_list

        self.observations = [
            'TOut.T', 'heaPum.COP', 'heaPum.COPCar', 'heaPum.P',
            'heaPum.QCon_flow', 'heaPum.QEva_flow', 'heaPum.TConAct',
            'heaPum.TEvaAct', 'preHea.Q_flow', 'rad.Q_flow', 'rad.m_flow',
            'sunHea.Q_flow', 'sunRad.y', 'temRet.T', 'temRoo.T', 'temSup.T',
            'weaBus.HDifHor', 'weaBus.HDirNor', 'weaBus.HGloHor',
            'weaBus.HHorIR', 'y', 'time']
        self.P = P
        self.SP_temp_night = SP_temp_night + Celcius_to_Kelvin
        self.SP_temp_day = SP_temp_day + Celcius_to_Kelvin
        self.nighttime_start = nighttime_start
        self.nighttime_end = nighttime_end

    def get_control(self, observation, hour=0):
        """Computes the control actions.

        Parameters
        ----------
        obs : dict
            Dict containing the temperature observations.
        hour : int
            Current hour in the simulation time.

        Returns
        -------
        controls : dict
            Dict containing the control inputs.
        """
        controls = {}
        if hour < self.nighttime_end or hour > self.nighttime_start:
            control_temp = self.SP_temp_night
        else:
            control_temp = self.SP_temp_day
        
        control_u = 0.0
        control_u = min(self.P * max(control_temp - observation, 0.0), 1.0)
        controls['u'] = [control_u]

        return controls


class Household:

    def __init__(
        self,
        SP_night,
        nighttime_end=6,
    ):
        
        self.SP_night = SP_night
        self.nighttime_end = nighttime_end
        

    def get_ApartTherm_kpis(self,**kwargs):
        weather = kwargs['weather'] #"CH_BS_Basel"
        num_sim_days = kwargs['sim_days']
        P, SP_day, nighttime_start = kwargs['params']
        SP_night_calvin = self.SP_night + Celcius_to_Kelvin
        lower_comfort = SP_night_calvin - 1.5
        upper_comfort = SP_night_calvin + 1.5
        my_kpi_options = {
                            "kpi1": {"name": "heaPum.P", "type": "avg"},
                            "kpi2": {"name": "temRoo.T","type": "avg_dev","target": [lower_comfort, upper_comfort],},
                            "kpi3": {"name": "temRoo.T","type": "tot_viol","target": [lower_comfort, upper_comfort],},
                            }
        env = energym.make(
            "SimpleHouseRad-v0", weather=weather, simulation_days=num_sim_days, kpi_options = my_kpi_options)
        mins_per_step = 5
        mins_in_a_day = 24 * 60
        steps_in_a_day = int(mins_in_a_day / mins_per_step)

        inputs = env.get_inputs_names()

        controller = PController(
                control_list=inputs, P=P, SP_temp_night= self.SP_night, SP_temp_day= SP_day,
                nighttime_start= nighttime_start, nighttime_end=self.nighttime_end
            )

        steps = steps_in_a_day * num_sim_days
        out_list = []
        outputs = env.get_output()['temRoo.T']
        hour = 0
        for i in range(steps):
            control = controller.get_control(outputs, hour)
            '''
            if i%12== 0:
                print(f'hour={env.get_date()}')
                print(control)
                print(env.step(control)['temRoo.T'] - Celcius_to_Kelvin)
            '''
            outputs = env.step(control)['temRoo.T'] 
            _,hour,_,_ = env.get_date()

            out_list.append(outputs - Celcius_to_Kelvin)
        
        avg_energy_hourly = []
        tot_energy_hourly = []
        room_temp = []
        avg_dev = []
        for h in range(24*num_sim_days):
            kpis = env.get_kpi((h*int(60/mins_per_step)),((h+1)*int(60/mins_per_step)-1))
            avg_energy_hourly.append(kpis['kpi1']['kpi']) # Heat pump: average consumed power (W)
            tot_energy_hourly.append(kpis['kpi1']['kpi']*int(60/mins_per_step))
            # Measure the average room temperature deviation during the evening and night time. 
            # Agents should learn the to shift early enough such that at 19 o'clock the temperature is at the nighttime set point 
            if h%24 < self.nighttime_end or h%24 > 19: 
                avg_dev.append(kpis['kpi2']['kpi']) # Room temperature: average deviation from comfort level
            help = out_list[h*int(60/mins_per_step):(h+1)*int(60/mins_per_step)]
            room_temp.append(np.sum(help)/int(60/mins_per_step))

        env.close()
        return avg_energy_hourly, tot_energy_hourly, room_temp, avg_dev



    def get_ApartElectricity_cost(self,**kwargs):
        """
        Electricity cost of one household is of the form:

        J_i(load_i, load_{-i}) = \sum_{h=1}^24 p(load_i^h,load_{-i}^h)*load_i^h

        with price function:
        
        p(load_i^h,load_{-i}^h) = rho_1^h*\sum_{i=1}^N load_i^h + rho_2^h,

        where 

        rho_1^h = 0.015$/kWh if h\in[0,6]U[10,18]U[22,24] and 0.3$/kWH at peak times
        rhh_2^h = 0.05$/kWh  if h\in[0,6]U[10,18]U[22,24] and 0.1$/kWH at peak times
                            
        """
        N = kwargs['num_agents']
        agent = kwargs['agent']
        sim_days = kwargs['sim_days']
        rho1 = 0.015 
        rho2 = 0.05 
        rho1_peak = rho1 * 2
        rho2_peak = rho2 * 2
        energy_hourly = kwargs['energy_hourly']

        tot_load_hourly = []
        cost = 0
        for h in range(24*sim_days):
            tot_load_hourly.append(np.sum([energy_hourly[i][h] for i in range(N)]))
            if (0 <= h%24 < 6) or (10 <= h%24 < 18) or (22 < h%24 <= 24):
                cost = cost + (rho1*tot_load_hourly[-1]*1/1000 + rho2) * energy_hourly[agent][h] * 1/1000
            else:
                cost = cost + (rho1_peak*tot_load_hourly[-1] * 1/1000 + rho2_peak) * energy_hourly[agent][h] * 1/1000 

        return cost
