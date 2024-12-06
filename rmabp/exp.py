import numpy as np
from .simulators import WIsim
from .solvers import *
from .utilities import Collection, whittle_index, steady_state_matrix
from .bandits import BanditPR, RMAB
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import json

except_undis = Exception('cannot calculate undiscounted non-relative values')


class ExpSettings:
    """Settings for an experiment.
    
    To be used in conjuction with an RMAB and RMABController in an Experiment.
    The experimental settings do not include settings for the RMAB or
    RMABController.
    """
    
    def __init__(self,
            K = 1,
            discount = 1,
            relative = True,
            T = 100,
            num_trials = 1):
        if (discount == 1) and not relative:
            raise except_undis
        self.relative = relative
        self.discount = discount
        self.K = K
        self.T = T
        self.num_trials = num_trials
        



class Experiment:
    """ Experimenting with an RMAB and RMABController.
    
    Based on the experimental settings, the experiment will use the
    RMABController to controll the RMAB in multiple simulations.
    """
    
    def __init__(self, rmab, controller, settings, seed = None):
        if seed is not None:
            np.random.seed(seed)
        self.rmab = rmab
        self.controller = controller
        self.settings = settings
        self.episodes = []
        self.wiepisodes = []
    
    def __repr__(self):
        return self.rmab.__repr__()+ self.controller.__repr__()+ self.settings.__repr__()
    
    def exp_total_reward(self):
        if len(self.episodes) == 1:
            m = self.episodes[0].rewards.sum(0)
            s = np.zeros(self.settings.T)
        else:
            rewards = np.zeros([self.settings.T,len(self.episodes)])
            for i, e in enumerate(self.episodes):
                rewards[:,i] = e.rewards.sum(0)
            m = rewards.mean(1)
            s = rewards.std(1)
        return(m, s)
    
    def exp_cum_reward(self):
        if len(self.episodes) == 1:
            m = self.episodes[0].rewards.sum(0).cumsum()
            s = np.zeros(self.settings.T)
        else:
            rewards = np.zeros([self.settings.T,len(self.episodes)])
            for i, e in enumerate(self.episodes):
                rewards[:,i] = e.rewards.sum(0).cumsum()
            m = rewards.mean(1)
            s = rewards.std(1)
        return(m, s)
    
    def exp_time_avg_reward(self):
        if len(self.episodes) == 1:
            m = self.episodes[0].rewards.sum(0).cumsum()
            s = np.zeros(self.settings.T)
        else:
            times = np.array(range(1,self.settings.T+1))
            rewards = np.zeros([self.settings.T,len(self.episodes)])
            for i, e in enumerate(self.episodes):
                rewards[:,i] = e.rewards.sum(0).cumsum()/times
            m = rewards.mean(1)
            s = rewards.std(1)
        return(m, s)
    
    def plot_total_reward(self):
        T = self.settings.T
        t = range(1,T+1)
        m, s = self.exp_total_reward()
        plt.plot(t,m)
        plt.fill_between(t, m-s, m+s, alpha = 0.2)
    
    def plot_cum_reward(self):
        T = self.settings.T
        t = range(1,T+1)
        m, s = self.exp_cum_reward()
        plt.plot(t,m)
        plt.fill_between(t, m-s, m+s, alpha = 0.2)
    
    def plot_exp_time_avg_reward(self, **kwargs):
        T = self.settings.T
        t = range(1,T+1)
        m, s = self.exp_time_avg_reward()
        plt.plot(t,m, **kwargs)
        plt.fill_between(t, m-s, m+s, alpha = 0.2)
        plt.xlabel("Time step")
        plt.ylabel("Time Average Total Reward")
    
    def execute(self, steady_dist = True):
        T = self.settings.T
        print("executing")
        for trial in range(self.settings.num_trials):
            self.rmab.randomise_states(steady_dist = steady_dist)
            self.controller.reset()
            episode = RMABEpisode(self.rmab.states(), T, self.rmab.identifiers())
            wiepisode = WhittleIndexEpisode(self.controller, T)
            WIsim(self.rmab, self.controller, T, episode, wiepisode)
            self.episodes.append(episode)
            self.wiepisodes.append(wiepisode)
    
    def tot_rewards(self):
        tot_rewards = [e.rewards.sum() for e in self.episodes]
        return [np.mean(tot_rewards), np.var(tot_rewards)]


class Experiments(list, Collection):
    """ List of Experiments, each combining an RMAB, controller and settings.
    
    A list subclass that can automatically generate combinations of RMABs,
    RMABControllers and ExpSettings.
    """
    
    def __init__(self, *args,
                 rmabs = None,
                 controllers = None,
                 settingss = None,
                 seed = None,
                 seedall = None,
                 controller_kwargs = {},
                 RMABargs = {}):
        
        if seed is not None:
            np.random.seed(seed)
        
        if None in [rmabs, controllers, settingss]:
            super().__init__(*args)
            self.ind = None
        else:
            arg = list()
            ind = list()
            for i, r in enumerate(rmabs):
                for j, c in enumerate(controllers):
                    for k, s in enumerate(settingss):
                        controller = homoRMABController(r, c, s.K, discount = s.discount, RMABargs = RMABargs, **controller_kwargs)
                        arg.append(Experiment(r, controller, s, seed = seedall))
                        ind.append([i,j,k])
            super().__init__(arg)
            self.ind = ind
    
    def execute(self, **kwargs):
        for e in self:
            e.execute(**kwargs)
    
    def tot_rewards(self):
        return np.stack([e.tot_rewards() for e in self])



def homogeneous_sim_setup(path, rmab, controller,
                          discount = 1, relative = True, T = 1000):
    """Prepares a directory for simulations.
    
    Saves details about the bandits and controller in the directory in a
    format that can be loaded again to run experiments.
    

    Parameters
    ----------
    path : str
        the path of the experiemnt directory. The path will be created if it
        does not already exist.
    rmab : RMAB
        homogenous restless multiarmed bandits.
    controller : RMABController
        the controller.
    discount : float, optional
        The discount factor. The default is 1.
    relative : Bool, optional
        Whether to use relative values or not. The default is True.
    T : int > 1, optional
        Simulation time. The default is 1000.

    Returns
    -------
    None.

    """
    
    # make directory if it does not exist
    if len(path) > 0:
        if not os.path.exists(path):
            os.makedirs(path)
    
    # make folder to save experiment paramters
    param_path = os.path.join(path, "experiment_parameters")
    if not os.path.exists(param_path):
        os.makedirs(param_path)
    
    # make folder to save simulations
    sim_path = os.path.join(path, "simulations")
    if not os.path.exists(sim_path):
        os.makedirs(sim_path)
    
    # first bandit contains same info as all because rmab is homogenous
    b = rmab[0]
    
    # save transition probabilites and reward functions
    b.save_PR(path = param_path)
    
    # save Whittle indices
    wi = whittle_index(b.P, b.R, discount = discount, relative = relative)
    np.savetxt(
        os.path.join(param_path,"wi_true.csv"),
        wi, delimiter = ","
    )
    # save state order of whittle indices
    np.savetxt(
        os.path.join(param_path,"wi_order_true.csv"),
        wi.argsort()[::-1], delimiter = ",", fmt='%i'
    )
    # save passive policy steady state distribution
    np.savetxt(
        os.path.join(param_path,"P0_steady_dist.csv"),
        b.steady_dist, delimiter = ","
    )
    
    # save random policy steady state distribution
    K = controller.K
    I = len(rmab)
    rand_steady = steady_state_matrix((I-K)/I*b.P[0,:,:] + K/I*b.P[1,:,:]).mean(0)
    np.savetxt(
        os.path.join(param_path,"rand_steady_dist.csv"),
        rand_steady, delimiter = ","
    )
    
    for c in controller:
        con = controller[c]
        break
    
    # save config json file
    config = {
        "bandit_ID" : b.identifier,
        "I" : I,
        "L" : b.L,
        "bandit_args" : b.bandit_args,
        "discount" : discount,
        "relative" : relative,
        "K" : K,
        "control_ratio" : K/I,
        "T" : T,
        "exp_pass_reward" : b.steady_dist @ b.R[0,:],
        "exp_rand_reward" : rand_steady @ ((I-K)*b.R[0,:] + K*b.R[1,:]),
        "controller_ID" : con.label,
        "controller_args" : con.controller_args,
        "p_explore0" : controller.p_explore(0),
        "wlr0" : con.wlr(0),
        "alpha0" : con.alpha(0)
    }
    
    with open(os.path.join(param_path, "config.json"), "w") as file:
        json.dump(config, file)
    
        
def generate_seeds(seed_path, num):
    rng = np.random.default_rng()
    new_seeds = rng.choice(9999999, size=num, replace=False)
    
    seed_file = os.path.join(seed_path, "seeds.csv")
    if os.path.isfile(seed_file):
        old_seeds = np.loadtxt(seed_file, delimiter=",", dtype = int)
        all_seeds = np.concatenate([old_seeds, new_seeds])
    else:
        all_seeds = new_seeds
    np.savetxt(seed_file, all_seeds, delimiter=",", fmt='%i')
    





class Simulation:
    def __init__(self, path = ""):
        self.path = path
        self.param_path = os.path.join(path, "experiment_parameters")
        self.sim_path = os.path.join(path, "simulations")
        
        # load config file
        with open(os.path.join(self.param_path, "config.json")) as file:
            config = json.load(file)
        
        # save config dictionary entires as variables
        for key, val in config.items():
            exec("self." + key + "=val")
        
        # generate homogenous RMAB
        b = BanditPR(
            self.L,
            identifier = self.bandit_ID,
            loadpath = self.param_path
        )
        self.rmab = RMAB([deepcopy(b) for i in range(self.I)])
        
        # dictionary to map controller names
        cont_names = {
            Cmodel.label : Cmodel,
            QWICAvrBor.label : QWICAvrBor,
            QWICladder.label : QWICladder,
            WIController.label : WIController,
            RandomController.label : RandomController
        }
        
        self.controller = homoRMABController(
            self.rmab,
            cont_names[self.controller_ID],
            self.K,
            #relative = self.relative,
            **self.controller_args,
            wlr = lambda t: self.wlr0,
            alpha = lambda t: self.alpha0,
            RMABargs = {"p_explore" : lambda t: self.p_explore0}
        )
        
        self.WIcont = homoRMABController(
            self.rmab,
            WIController,
            self.K,
            relative = self.relative,
            bandit = self.rmab[0],
            wlr = lambda t: self.wlr0,
            alpha = lambda t: self.alpha0,
            RMABargs = {"p_explore" : lambda t: self.p_explore0}
        )
        self.WIcont.reset()
        
        
    
    
    def save_WIC_warmup(self, warmT = 1000, restT = 10000, seed = None):
        if seed is not None:
            np.random.seed(seed)
        # begin a warmup from the passive action steady state distribution
        # using the whittle index controller with exploration
        self.rmab.randomise_states(steady_dist = True)
        episode = RMABEpisode(
            self.rmab.states(), 
            warmT,
            self.rmab.identifiers()
        )
        wiepisode = WhittleIndexEpisode(self.WIcont, warmT)
        WIsim(self.rmab, self.WIcont, warmT, episode, wiepisode)
        
        # run a simulation for restT using the whittle index controller
        # with exploration
        episode = RMABEpisode(
            self.rmab.states(), 
            restT,
            self.rmab.identifiers()
        )
        wiepisode = WhittleIndexEpisode(self.WIcont, restT)
        WIsim(self.rmab, self.WIcont, restT, episode, wiepisode)
        
        # make folder to save Whittle index controlled data
        WIC_path = os.path.join(self.path, "WIC_data")
        if not os.path.exists(WIC_path):
            os.makedirs(WIC_path)
        
        np.savetxt(
            os.path.join(WIC_path, "states.csv"),
            episode.states,
            fmt='%i',
            delimiter = ","
        )
        
        total_rewards = episode.rewards.sum(0)
        
        # save config json file
        WIC_data = {
            "warmT" : warmT,
            "restT" : restT,
            "seed" : seed,
            "reward_expected" : total_rewards.mean(),
            "reward_deviation" : total_rewards.std()
        }
        
        with open(os.path.join(WIC_path, "WIC_data.json"), "w") as file:
            json.dump(WIC_data, file)
    
    
    def run(self,
            init_dist = "WICWE",
            reset = True
            ):
        
        init_dist_options = ["WICWE", "passive", "uniform"]
        
        if init_dist == "passive":
            self.rmab.randomise_states(steady_dist = True)
        elif init_dist == "uniform":
            self.rmab.randomise_states(steady_dist = False)
        elif init_dist == "WICWE": #whittle index controller with exploration
        
            initial_state = np.loadtxt(
                os.path.join(self.path, "WIC_data", "states.csv"),
                delimiter=",", dtype = int)
            ind = np.random.choice(initial_state.shape[1])
            initial_state = initial_state[:,ind]
            self.rmab.set_states(initial_state)
        else:
            raise Exception("unknown initial distribution")
        
        if reset:        
            self.controller.reset()
        
        episode = RMABEpisode(
            self.rmab.states(), 
            self.T,
            self.rmab.identifiers()
        )
        wiepisode = WhittleIndexEpisode(self.controller, self.T)
        WIsim(self.rmab, self.controller, self.T, episode, wiepisode)
        
        return episode, wiepisode
    
    def run_trials(self):
        seed_path = os.path.join(self.param_path, "seeds.csv")
        seeds = np.loadtxt(seed_path, delimiter=",", dtype = int)
        trials = len(seeds)
        
        trial0 = len([x for x in os.listdir(self.sim_path) if "trial" in x])
        
        for trial in range(trial0, trials):
            
            np.random.seed(seeds[trial])
            episode, wiepisode = self.run()
            
            
            path_trial = os.path.join(self.sim_path, "trial"+str(trial))
            os.mkdir(path_trial)
            
            np.savetxt(
                os.path.join(path_trial, "rewards_total.csv"),
                episode.rewards.sum(0),
                delimiter = ","
            )
            np.savetxt(
                os.path.join(path_trial, "rewards_running_avg.csv"),
                episode.rewards.sum(0).cumsum()/np.arange(1,1+episode.rewards.shape[1]),
                delimiter = ","
            )
            for ident in wiepisode.wi:
                path_trial_wi = os.path.join(path_trial, ident)
                os.mkdir(path_trial_wi)
                np.savetxt(
                    path_trial_wi + "\\wi.csv",
                    wiepisode.wi[ident],
                    delimiter = ","
                )
                np.savetxt(
                    path_trial_wi + "\\wiorder.csv",
                    wiepisode.wiorder[ident],
                    delimiter = ",",
                    fmt='%i'
                )


