import ConfigParser
import os

# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class Params(object):
    __instance = None
    def __new__(cls,configFile=None):
        if Params.__instance is None:
            Params.__instance = object.__new__(cls)



            config = ConfigParser.ConfigParser()
            config.read(configFile)

            # Define Simulation params
            Params.__instance.max_timesteps  = config.getint('sim','max_timesteps')
            Params.__instance.plot_sim       = config.getboolean('sim','plot_sim')
            Params.__instance.live_plots     = config.getboolean('sim','live_plots')
            Params.__instance.plot_cbf       = config.getboolean('sim','plot_cbf')
            Params.__instance.plot_constrs   = config.getboolean('sim','plot_constrs')
            Params.__instance.plot_clf       = config.getboolean('sim','plot_clf')
            Params.__instance.plot_delta     = config.getboolean('sim','plot_delta')
            Params.__instance.p_cbf          = config.getfloat('sim','p_cbf')
            
            # Decentralized True will make CBF ij and ji
            # Assigns priority to agents for better deconfliction
            Params.__instance.decentralized  = config.getboolean('sim','decentralized')

            # CLF Fields
            Params.__instance.epsilon = config.getfloat('clf','epsilon')
            Params.__instance.p = config.getfloat('clf','p')
            Params.__instance.gamma = config.getfloat('clf','gamma')

            # Dynamics Params
            Params.__instance.step_size = config.getfloat('dynamics','step_size') # seconds

            # Unicycle Dynamics Params
            Params.__instance.v_upper_bound = config.getfloat('unicycle','v_upper_bound')
            Params.__instance.w_upper_bound = config.getfloat('unicycle','w_upper_bound')
            Params.__instance.vel_penalty = config.getfloat('unicycle','vel_penalty')
            Params.__instance.steer_penalty = config.getfloat('unicycle','steer_penalty')
            Params.__instance.l = config.getfloat('unicycle','l')

            # Unicycle extended Dynamics
            Params.__instance.we_upper_bound = config.getfloat('unicycle_extended', 'we_upper_bound')
            Params.__instance.mu_upper_bound = config.getfloat('unicycle_extended', 'mu_upper_bound')
            Params.__instance.acc_penalty = config.getfloat('unicycle_extended', 'acc_penalty')
            Params.__instance.steere_penalty = config.getfloat('unicycle_extended', 'steere_penalty')

            # Single Integrator Dynamics
            Params.__instance.max_speed = config.getfloat('single_int','max_speed')

            # Double Integrator Dynamics
            Params.__instance.max_accel = config.getfloat('double_int','max_accel')


        return Params.__instance
    