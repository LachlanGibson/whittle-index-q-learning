
import numpy as np
import matplotlib.pyplot as plt
from .utilities import whittle_index, Collection





class BanditController:
    '''
    General superclass that contains attributes and methods related controlling
    a specific type of bandit.
    '''
    
    label = "Controller Template"
    
    def __init__(self, identifier, L, A = 2, discount = 0.99, **kwargs):
        self.identifier = identifier
        # string - the unique string that identifies the bandit type
        
        self.L = L
        # int > 0 - the number of states available to the bandit
        
        self.A = A
        # int > 0 - the number of actions (viz. 2)
        
        self.discount = discount
        # 0 <= float <= 1 - the discount factor
        
        self.counts = None
        # (A,L,L) int array - counts of the observed transitions.
        # If counts[0,5,3] is 34, then state 5 has transitioned to state 3
        # under action 0 34 times.
        
        self.nu = None
        # (A,L) int array - counts of observed state-action pairs.
        # If counts[0,3] is 12, then the passive action has been taken in
        # in state 3 12 times.
        
        self.Rtotal = None
        # (A,L) float array - the total rewards observed for each action state
        # pair. If Rtotal[1,3] is 2.58, then the total reward observed from
        # state 3 under action 1 is 2.58.
        
        self.R = None
        # (A,L) float array - the estimated averaged rewards observed for
        # each action state pair.
        
        self.P = None
        # (A,L,L) float array - the estimated transition matrices.
        
        self.Qun = None
        # (L,L,A) float array - Q values at each Whittle index (unordered)
        
        self.Qor = None
        # (L,L,A) float array - Q values at each Whittle index (ordered)
        
        self.Qi = None
        self.Hi = None
        # (L+1,L,A) float arrays - Decomposed quality functions. The first
        # index distinguishes the policy. Qi[3,6,1] is the quality under policy
        # 3 at state 6 and action 1. Hi represents  the expected (discounted)
        # total number of taxes paid.
        
        self.wi = None
        # (L,) float array - Whittle index estimates at each state
        
        self.wiorder = None
        # (L,) int array - ordering of states from maximum whittle index
        
        self.dQdH = None
        # (L+1,L) float array - (Qi[:,:,1]-Qi[:,:,0])/(Hi[:,:,1]-Hi[:,:,0])
        # with smoothing based on the whittle index learning rate wlr.
        
        self.controller_args = dict()
        # dictionary - store controller specific keyword arguments (excluding
        # functions such as learning rates)
        
        self.alpha = self.alpha_default
        # function - Q learning rate
        
        self.wlr = self.wlr_default
        # function - whittle index learning rate
    
    
    def alpha_default(self, t):
        return 1.0
    
    def wlr_default(self, t):
        return 1.0
    
    def reset(self):
        self.counts = None
        self.nu = None
        self.Rtotal = None
        self.R = None
        self.P = None
        self.Qun = None
        self.Qor = None
        self.Qi = None
        self.Hi = None
        self.wi = None
        self.wiorder = None
        self.dQdH = None
    
    def update(self, *args, **kwargs):
        pass
    
    def normalise_counts(self, counts = None, Preg = None, PregI = None):
        if counts is None:
            counts = self.counts
        if Preg is None:
            Preg = self.Preg
        if PregI is None:
            PregI = self.PregI
        L = counts.shape[1]
        Pcreg = counts+Preg+PregI*np.stack([np.eye(L),np.eye(L)])
        return Pcreg/np.expand_dims(Pcreg.sum(-1),-1)
    
    def maxR(self, Rtotal, c):
        div = np.divide(Rtotal, c,
                        out = np.full(Rtotal.shape, np.NINF), where = c!=0)
        r_max = div.max()
        if r_max == np.NINF:
            return 0.0
        else:
            return r_max
    
    def average_reward(self, Rtotal = None, counts = None,
                       Rreg = None, r_max = None):
        if counts is None:
            counts = self.counts
        c = counts.sum(-1)
        if Rtotal is None:
            Rtotal = self.Rtotal
        if Rreg is None:
            Rreg = self.Rreg
        if r_max is None:
            r_max = self.maxR(Rtotal, c)
            
        return (Rtotal+Rreg*r_max)/(Rreg+c)
    
    def region_policies(self, wiorder = None):
        if wiorder is None:
            wiorder = self.wiorder
        rpolicy = np.zeros([self.L+1, self.L], dtype = int)
        for i in range(1, self.L+1):
            rpolicy[i, wiorder[:i]] = 1
        return rpolicy
    


class RMABController(dict, Collection):
    '''
    General superclass that is a dictionary of BanditController. It contains
    attributes and methods related to controlling a RMAB.
    
    Methods
    states_wi - returns the Whittle index value of each bandit
        args:
        states - list of length I of each bandit state (int)
        identifiers - list of bandit identifiers. Same length as states
        wi = None - dictionary of Whittle index values (defaults to self.wi)
        returns:
            ind - (I,) float array - the Whittle index of each bandit
    '''
    def __init__(self, controllers, K, identifiers, p_explore = None):
        if isinstance(controllers, list):
            controllers = {c.identifier : c for c in controllers}
        super().__init__(controllers)
        if p_explore is None:
            p_explore = self.p_explore_default
        self.p_explore = p_explore
        self.K = K
        self.identifiers = identifiers
        self.explore = True
    
    def __get_wi(self):
        return {c : self[c].wi for c in self}
    
    def __set_wi(self, wi):
        for c in self:
            self[c].wi = wi[c]
    
    wi = property(__get_wi, __set_wi)
    
    def __get_wiorder(self):
        return {c : self[c].wiorder for c in self}
    
    def __set_wiorder(self, wiorder):
        for c in self:
            self[c].wiorder = wiorder[c]
    
    wiorder = property(__get_wiorder, __set_wiorder)
    
    def states_wi(self, states, wi = None):
        if wi is None:
            wi = self.wi
        ind = np.zeros(len(states))
        for i, ID in enumerate(self.identifiers):
            ind[i] = wi[ID][states[i]]
        return ind
    
    def policy_wi(self, states):
        # Current Whittle index of each bandit
        ind = self.states_wi(states)
        # array indices of ordered unique Whittle indices
        _, s = np.unique(ind, return_inverse = True)
        # sort indices, adding noise to break ties
        s = (s + np.random.rand(*s.shape)).argsort()
        actions = [0 for i in range(len(states))]
        for i in s[-self.K:]:
            actions[i] = 1
        return actions
    
    def policy(self, states):
        actions = self.policy_wi(states)
        if self.explore:
            actions = np.random.permutation(actions).tolist()
        return actions
    
    def reset(self):
        self.explore = True
        for c in self:
            self[c].reset()
    
    def update(self, episode):
        episode_decom = episode.decompose_types()
        for c in self:
            self[c].update(episode_decom[c])
        
        # decide whether or not to explore next step
        if self.p_explore(episode.t) > np.random.rand():
            self.explore = True
        else:
            self.explore = False
    
    def p_explore_default(self, n):
        return n**(-1/2)
    

def homoRMABController(rmab, Controller, K, RMABargs = {},  *args, **kwargs):
    sa = rmab.num_state_actions()
    controllers = {
        ID : Controller(ID, sa[ID][0], *args, A = sa[ID][1], **kwargs)
        for ID in sa}
    return RMABController(controllers, K, rmab.identifiers(), **RMABargs)




class RMABEpisode:
    '''
    This class stores episode data of a restless multiarmed bandit (RMAB).
    The episode is intialised with a list of initial states 'states0' and the
    total number of time steps 'T'.
    
    Attributes
    T - int - Total number of time steps, including steps not yet taken.
    I - int - Total number of bandits/projects.
    states - (I,T+1) array - Recorded states of each project at each time step.
    rewards - (I,T) array - Recorded rewards of each project at each time step.
    actions - (I,T) array - Recorded actions of each project at each time step.
    t - int - the current time step.
    types_ind - dictionary - keys are identifiers and values are lists of ind
    homogeneous - boolean - True if all bandits are the same type
    
    Methods
    step - sets the states, rewards and actions attributes at step t to
           s1, r and a respectively. Increases t by one.
    add_time - Increases the episode length by DT.
    decompose_types_ind - identify indices of bandits of each type
    run_avg_reward - returns running time average total reward
    plot_avg_reward - plots the running time average total reward.
    plot_state_hist - plots a histogram of the project states in the FinalT
                      time steps.
    '''
    def __init__(self, states0, T, identifiers,
                 states = None, rewards = None, actions = None, t = None):
        self.T = T
        self.I = len(states0)
        if isinstance(identifiers, list):
            self.identifiers = identifiers
        else:
            self.identifiers = [identifiers for i in range(self.I)]
        if states is None:
            self.states = np.zeros([self.I,T+1], dtype = int)
            self.states[:,0] = states0
        else:
            self.states = states
        if rewards is None:
            self.rewards = np.zeros([self.I,T])
        else:
            self.rewards = rewards
        if actions is None:
            self.actions = np.zeros([self.I,T], dtype = int)
        else:
            self.actions = actions
        if t is None:
            self.t = 0
        else:
            self.t = t
        self.types_ind = self.decompose_types_ind()
        self.homogeneous = (len(self.types_ind) == 1)
    
    def add_time(self, DT):
        states  = np.zeros([self.I, self.T + DT + 1], dtype = int)
        rewards = np.zeros([self.I, self.T + DT])
        actions = np.zeros([self.I, self.T + DT], dtype = int)
        states[:,:self.T+1] = self.states
        rewards[:,:self.T] = self.rewards
        actions[:,:self.T] = self.actions
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.T += DT
    
    def step(self, s1, r, a):
        self.states[ :, self.t+1] = s1
        self.rewards[:, self.t  ] = r
        self.actions[:, self.t  ] = a
        self.t += 1
    
    def decompose_types_ind(self, identifiers = None):
        if identifiers is None:
            identifiers = self.identifiers
        types_ind = {}
        for i, ID in enumerate(identifiers):
            if ID not in types_ind:
                types_ind[ID] = [i]
            else:
                types_ind[ID].append(i)
        return types_ind
    
    def decompose_types(self):
        episode_decom = {}
        for ID in self.types_ind:
            ind = self.types_ind[ID]
            episode_decom[ID] = RMABEpisode(
                self.states[ind,0], self.T, ID,
                states  = self.states[ind,:],
                rewards = self.rewards[ind,:],
                actions = self.actions[ind,:],
                t = self.t
                )
        return episode_decom
    
    def run_avg_reward(self):
        return self.rewards.sum(0).cumsum()/np.arange(1,self.T+1)
    
    def plot_run_avg_reward(self, **kwargs):
        times = np.arange(1,self.T+1)
        plt.plot(times, self.run_avg_reward(), **kwargs)
        plt.xlabel('Time Step')
        plt.ylabel('Running Time Average Total Reward')
    
    def plot_total_reward(self, mawindow = 1, perarm = False, **kwargs):
        times = np.arange(mawindow, self.T+1)
        total = self.rewards.sum(0)
        ma = total.cumsum()
        ma[mawindow:] = ma[mawindow:] - ma[:-mawindow]
        ma = ma[mawindow-1:]/mawindow
        if perarm:
            ma = ma/self.I
        plt.plot(times, ma, **kwargs)
        plt.xlabel('Time Step')
        if perarm:
            ylab = 'Average Reward per Arm'
        else:
            ylab = 'Total Reward'
        plt.ylabel(ylab)
    
    def plot_state_hist(self, FinalT = 1, **kwargs):
        plt.hist(self.states[:,-FinalT:].flatten(), **kwargs)
        plt.xlabel('State')
        plt.ylabel('Number of Projects')


class WhittleIndexEpisode:
    '''
    
    '''
    def __init__(self, rmabcon, T):
        self.T = T
        self.wi = {c : np.zeros([rmabcon[c].L, T+1]) for c in rmabcon}
        self.wiorder = {c : np.zeros([rmabcon[c].L, T+1]) for c in rmabcon}
        self.t = 0
    
    def step(self, rmabcon):
        wi = rmabcon.wi
        wiorder = rmabcon.wiorder
        for c in rmabcon:
            self.wi[c][:,self.t] = wi[c]
            self.wiorder[c][:,self.t] = wiorder[c]
        self.t += 1
    
    def plot_wi(self, true_wi = None, returnfig = False):
        figs = {}
        colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for c in self.wi:
            figs[c] = plt.figure()
            if true_wi is not None:
                #plt.hlines(true_wi[c],1, self.T,
                #           linestyles = 'dashed', label = 'True')
                for i, wit in enumerate(true_wi[c]):
                    line_label = "State " + str(i+1) + " index = " + str(wit)
                    plt.hlines(wit,0,1+self.T, linestyle='dashed', color = colours[i%len(colours)], label = line_label)
            plt.plot(self.wi[c].transpose())
            plt.xlabel('Time Step')
            plt.ylabel('Estimated Whittle Index')
            plt.title(c)
        if returnfig:
            return figs


    

class Cmodel(BanditController):
    '''A model-based Whittle Index Controller
    
    This controller learns the transition matricies and expected reward
    functions by estimating them using (regularised) averages. The Whittle
    indices are estimated using the estimated transition probabilities and
    expected rewards. The controller explores with some probability gamma(t),
    otherwise employes the Whittle index policy based on its estimates of the
    Whittle indices.
    
    Subclass Attributes
    Preg - float or int - P regularisation: number of extra counts to be added
                          to each transition
    Rreg - float or int - R regularisation: number of extra counts of r_max
                          reward to be added to each state-action pair
    
    Subclass Methods
    normalise_counts - returns an estimate of P from counts
    maxR - returns the maximum expected reward
    average_reward - returns an estimate of R from Rtotal and counts
    update - updates controller from new data
    '''
    
    label = "Model"
    
    def __init__(self, *args, Preg = 1, PregI = 0, Rreg = 1,**kwargs):
        super().__init__(*args, **kwargs)
        self.controller_args = {"Preg" : Preg, "PregI" : PregI, "Rreg" : Rreg}
        self.Preg = Preg
        self.PregI = PregI
        self.Rreg = Rreg
        self.Rraw = np.zeros([2, self.L])
        self.reset()
    
    def reset(self):
        super().reset()
        L = self.L
        self.Rraw = np.zeros([2, L])
        self.r_max = 0.0
        self.counts = np.zeros([2,L,L], dtype = int)
        self.P = self.normalise_counts()
        self.Rtotal = np.zeros([2,L])
        self.R = self.average_reward(r_max = 0.0)
        self.wi = whittle_index(self.P, self.R, discount = self.discount)
        self.wiorder = self.wi.argsort()[::-1]
    
    def update(self, episode):
        '''
        x0 - list - list of previous states
        x1 - list - list of subsequent states
        a - list - list of actions
        r - list - list of rewards
        t - int - the current time step
        '''
        t = episode.t - 1
        x0 = episode.states[:,t]
        x1 = episode.states[:,t + 1]
        a = episode.actions[:,t]
        r = episode.rewards[:,t]
        
        for i in range(episode.I):
            self.counts[a[i],x0[i],x1[i]] += 1
            self.Rtotal[a[i],x0[i]] += r[i]
        self.P = self.normalise_counts()
        self.R = self.average_reward()
        
        self.wi = whittle_index(self.P, self.R, discount = self.discount)
        self.wiorder = self.wi.argsort()[::-1]


class QWICAvrBor(BanditController):
    '''
    Avrachenkov, K.E., Borkar, V.S. (2021).
    Whittle index based Q-learning for restless bandits with average reward.
    https://doi.org/10.48550/arXiv.2004.14427 
    '''
    
    label = "AvrBor"
    
    def __init__(self, *args, wlr = None, alpha = None, **kwargs):
        super().__init__(*args, **kwargs)
        if wlr is None:
            wlr = self.wlr_default
        if alpha is None:
            alpha = self.alpha_default
        self.wlr = wlr
        self.alpha = alpha
        self.reset()
    
    def reset(self):
        super().reset()
        L = self.L
        self.Qun = np.zeros([L,L,self.A])
        self.counts = np.zeros([self.A,L,L], dtype = int)
        self.Rtotal = np.zeros([self.A,L])
        self.wi = np.zeros(L)
        self.wiorder = np.arange(L)
        self.nu = np.zeros([2,L], dtype = int)
    
    def alpha_default(self, t, C = 0.1):
        return C/np.ceil(t/500)
    
    def wlr_default(self, t, C = 0.1):
        if t == 0:
            return C
        else:
            return C/(1.0 + np.ceil(t*np.log(t)/500)) #(t % self.I == 0)*
    
    def update(self, episode):
        t = episode.t - 1
        discount = self.discount
        f = self.Qun.mean((1,2))
        wi = self.wi
        # update Q values
        for i in range(episode.I):
            x0 = episode.states[i,t]
            x1 = episode.states[i,t + 1]
            a = episode.actions[i,t]
            r = episode.rewards[i,t]
            self.Rtotal[a,x0] += r
            self.nu[a,x0] += 1
            lr = self.alpha(self.nu[a,x0])
            Q1 = self.Qun[:,x1,:].max(-1)
            DQ = lr*(r - a*wi + discount*Q1 - f - self.Qun[:,x0,a])
            self.Qun[:,x0,a] += DQ
            f += DQ/(2*self.L)
        
        # update Whittle index estimates
        ind = range(self.L)
        self.wi += self.wlr(t+1)*(self.Qun[ind,ind,1] - self.Qun[ind,ind,0])
        self.wiorder = self.wi.argsort()[::-1]



class QWICladder(BanditController):
    '''
    Gibson, L.J., Jacko, P., Nazarathy, Y. (2021).
    A Novel Implementation of Q-Learning for the Whittle Index.
    https://doi.org/10.1007/978-3-030-92511-6_10
    '''
    
    label = "Ladder"
    
    def __init__(self, *args, relative = True, alpha = None, wlr = None,
                 local_clock = False, wiclamp = True, medianbuffer = 1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.local_clock = local_clock
        self.controller_args = {
            "local_clock" : local_clock,
            "relative" : relative,
            "wiclamp" : wiclamp,
            "medianbuffer" : medianbuffer
        }
        self.relative = relative
        self.wiclamp = wiclamp
        self.medianbuffer = medianbuffer
        self.bufferind = 0
        self.buffer = np.full([self.medianbuffer,self.L+1,self.L], np.nan)
        if wlr is None:
            wlr = self.wlr_default
        if alpha is None:
            alpha = self.alpha_default
        self.wlr = wlr
        self.alpha = alpha
        self.reset()
    
    def reset(self):
        # Qi and Hi are initialised to be independent of state (uniform
        # transition probabilities). This result is exact for fully passive
        # or fully active policies and linearly interpolates between.
        super().reset()
        L = self.L
        dis = self.discount
        self.Qi = np.zeros([L+1,L,2])
        self.Hi = np.zeros([L+1,L,2])
        if self.relative:
            for i in range(L+1):
                self.Hi[i,:,0] = (2*dis*i/L-1)/(2*(2-dis))
        else:
            for i in range(L+1):
                self.Hi[i,:,0] = (i/L)*dis/(1-dis)
        self.Hi[:,:,1] = self.Hi[:,:,0] + 1
        self.dQdH = np.zeros([L+1,L])
        self.wi = np.zeros(L)
        self.wiorder = np.arange(L)
        if self.local_clock:
            self.nu = np.zeros([2,L], dtype = int)
        self.bufferind = 0
        self.buffer = np.full([self.medianbuffer,L+1,L], np.nan)
    
    def whittle_indices(self, dQdH = None):
        if dQdH is None:
            dQdH = self.dQdH
        
        wiorder = np.full(self.L, -1)
        wi = np.zeros(self.L)
        j = 0
        jend = self.L - 1
        while j <= jend:
            wituple = (np.NINF, -1)
            for s in range(self.L):
                if s not in wiorder:
                    candidate = dQdH[j,s]
                    wituple = max(wituple, (candidate, s))
            wiorder[j] = wituple[1]
            if j == 0:
                wi[wituple[1]] = wituple[0]
            else:
                if self.wiclamp:
                    wi[wituple[1]] = min(wituple[0], wi[wiorder[:j]].min())
                else:
                    wi[wituple[1]] = wituple[0]
            j += 1
            if jend >= j:
                wituple = (np.inf, -1)
                for s in range(self.L):
                    if s not in wiorder:
                        candidate = dQdH[jend+1,s]
                        wituple = min(wituple, (candidate, s))
                wiorder[jend] = wituple[1]
                if jend == self.L - 1:
                    wi[wituple[1]] = wituple[0]
                else:
                    if self.wiclamp:
                        wi[wituple[1]] = max(wituple[0], wi[wiorder[jend+1:]].max())
                    else:
                        wi[wituple[1]] = wituple[0]
                jend -= 1
        return wi, wiorder
    
    def alpha_default(self, t):
        return 1.0/t
    
    def wlr_default(self, t):
        return 1
    
    def update(self, episode):
        t = episode.t - 1
        discount = self.discount
        rpolicy = self.region_policies()
        
        if self.relative:
            fQ = self.Qi.mean((1,2))
            fH = self.Hi.mean((1,2))
        else:
            fQ = np.zeros(self.L+1)
            fH = np.zeros(self.L+1)
        
        
        for i in range(episode.I):
            x0 = episode.states[i,t]
            x1 = episode.states[i,t + 1]
            a = episode.actions[i,t]
            r = episode.rewards[i,t]
            
            for j in range(self.L+1):
                
                if self.local_clock:
                    self.nu[a,x0] += 1
                    lr = self.alpha(self.nu[a,x0])
                else:
                    lr = self.alpha(t+1)
                
                p = tuple(rpolicy[j,:])
                policy_not_constant = sum(p) not in [0, self.L]
                
                a1 = p[x1]
                
                DQ = lr*(r + discount*self.Qi[j,x1,a1]-fQ[j]-self.Qi[j,x0,a])
                self.Qi[j,x0,a] += DQ
                if self.relative:
                    fQ[j] += DQ/(2*self.L)
                if policy_not_constant:
                    DH = lr*(a + discount*self.Hi[j,x1,a1]-fH[j]-self.Hi[j,x0,a])
                    self.Hi[j,x0,a] += DH
                    if self.relative:
                        fH[j] += DH/(2*self.L)
        
        dQi = self.Qi[:,:,1] - self.Qi[:,:,0]
        dHi = self.Hi[:,:,1] - self.Hi[:,:,0]
        
        self.buffer[self.bufferind,:,:] = np.divide(dQi, dHi, out = dQi.copy(), where = dHi!=0)
        self.bufferind = (self.bufferind + 1)%self.medianbuffer
        dQdH = np.nanmedian(self.buffer, axis = 0)
        
        self.dQdH = self.dQdH + self.wlr(t)*(dQdH-self.dQdH)
        
        self.wi, self.wiorder = self.whittle_indices()
            
    
class WIController(BanditController):
    
    
    label = "Whittle Index Controller"
    
    def __init__(self, *args, bandit = None, alpha = None, relative = True, discount = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.relative = relative
        self.discount = discount
        if alpha is None:
            alpha = self.alpha_default
        self.alpha = alpha
        self.bandit = bandit
        self.reset()
    
    def reset(self):
        super().reset()
        self.wi = whittle_index(
            self.bandit.P,
            self.bandit.R,
            discount = self.discount,
            relative = self.relative)
        self.wiorder = np.arange(self.L)
    
    def alpha_default(self, t):
        return 1.0/t


class RandomController(BanditController):
    
    label = "Random Controller"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()
    
    def reset(self):
        super().reset()
        self.wi = np.random.rand(self.L)
        self.wiorder = np.arange(self.L)
    
    def update(self, episode):
        self.wi = np.random.rand(self.L)
        self.wiorder = self.wi.argsort()[::-1]





class QWICAvrBorAdaptive(BanditController):
    '''A version of Avrachenkov & Borkar with dynamic learning rates
    
    This controller 
    '''
    
    label = "AvrBorDLR"
    
    def __init__(self, *args, wlr = None, alpha = None, Hbound = [0.5, 2.0], **kwargs):
        super().__init__(*args, **kwargs)
        self.controller_args = {"Hbound" : Hbound}
        self.Hbound = Hbound
        if wlr is None:
            wlr = self.wlr_default
        if alpha is None:
            alpha = self.alpha_default
        self.wlr = wlr
        self.alpha = alpha
        self.reset()
    
    def reset(self):
        super().reset()
        L = self.L
        dis = self.discount
        self.Qun = np.zeros([L,L,self.A])
        self.counts = np.zeros([self.A,L,L], dtype = int)
        self.Rtotal = np.zeros([self.A,L])
        self.wi = np.zeros(L)
        self.wiorder = np.arange(L)
        self.nu = np.zeros([2,L], dtype = int)
        self.Hi = np.zeros([L+1,L,2])
        for i in range(L+1):
            self.Hi[i,:,0] = (2*dis*i/L-1)/(2*(2-dis))
        self.Hi[:,:,1] = self.Hi[:,:,0] + 1
    
    def alpha_default(self, t, C = 0.1):
        return C/np.ceil(t/500)
    
    def wlr_default(self, t, C = 0.1):
        if t == 0:
            return C
        else:
            return C/(1.0 + np.ceil(t*np.log(t)/500)) #(t % self.I == 0)*
    
    def update(self, episode):
        t = episode.t - 1
        discount = self.discount
        f = self.Qun.mean((1,2))
        fH= self.Hi.mean((1,2))
        wi = self.wi
        rpolicy = self.region_policies()
        # update Q values
        for i in range(episode.I):
            x0 = episode.states[i,t]
            x1 = episode.states[i,t + 1]
            a = episode.actions[i,t]
            r = episode.rewards[i,t]
            self.Rtotal[a,x0] += r
            self.nu[a,x0] += 1
            lr = self.alpha(self.nu[a,x0])
            Q1 = self.Qun[:,x1,:].max(-1)
            DQ = lr*(r - a*wi + discount*Q1 - f - self.Qun[:,x0,a])
            self.Qun[:,x0,a] += DQ
            f += DQ/(2*self.L)
            
            H1 = self.Hi[range(self.L+1),x1,rpolicy[:,x1]]
            DH = lr*(a + discount*H1 - fH - self.Hi[:,x0,a])
            self.Hi[:,x0,a] += DH
            fH += DH/(2*self.L)
        
        # update Whittle index estimates
        wlr = self.wlr(t+1)
        indcon = np.argsort(self.wiorder)
        for i in range(self.L):
            QmQ = self.Qun[i,i,1] - self.Qun[i,i,0]
            Hind = indcon[i] + (QmQ > 0)
            HmH = self.Hi[Hind,i,1] - self.Hi[Hind,i,0]
            HmH = max(min(self.Hbound[1],HmH),self.Hbound[0]) # clamp
            self.wi[i] += HmH*wlr*QmQ
        
        self.wiorder = self.wi.argsort()[::-1]







