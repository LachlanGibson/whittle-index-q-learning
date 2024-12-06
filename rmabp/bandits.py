import numpy as np
import os
from .utilities import steady_state_matrix, whittle_index, random_irreducible_P, indexability_bound, Collection

class RMAB(list, Collection):
    '''
    List subclass to contain a restless multi-armed bandit.
    constructor input is a list of bandits (of type Bandit)
    
    methods
    transition
    takes a list of actions of the same length as the list of bandits
    also takes a boolean return_rewards (True by default) which decides if
    the rewards should be returned the function calls the transition method
    of each bandit and creates a list of the rewards returns the list of
    rewards of each bandit if return_rewards is true
    
    states
    returns a list of the state of each bandit
    
    set_states
    set the states of the bandits as an inputed list-like object
    
    identifiers
    returns a list of the identifiers of each bandit
    
    num_state_actions
    returns a dictionary of tuples containing the number of states and number
    of actions in each type of bandit. The bandit identifiers are the keys.
    
    randomise_states
    randomises the states of all bandits by calling their random_state methods
    
    random_actions
    returns a list of random actions
    
    get_bandits
    gets the bandits  with identifiers in the list 'identifiers'.
    If first is true then only the first bandit is returned.
    Otherwise the whole list (as a new RMAB instance) is returned.
    
    get_wi
    gets a dictionary of Whittle indices for each type of bandit.
    '''
    def __init__(self, bandits = None):
        if bandits is None:
            bandits = []
        super().__init__(bandits)
    
    def transition(self, actions = None, return_rewards = True):
        if actions is None:
            actions = self.random_actions()
        if len(actions) != len(self):
            raise Exception('The number of actions ({}) does not equal the number of bandits ({}).'.format(len(actions), len(self)))
        rewards = list()
        for i, b in enumerate(self):
            rewards.append(b.transition(actions[i]))
        if return_rewards:
            return rewards
    
    def states(self):
        return [b.state for b in self]
    
    def set_states(self, states):
        for i, b in enumerate(self):
            b.state = states[i]
    
    def identifiers(self, unique = False):
        if unique:
            ident = list()
            for b in self:
                if b.identifier not in ident:
                    ident.append(b.identifier)
        else:
            ident = [b.identifier for b in self]
        return ident
    
    def num_state_actions(self):
        state_actions = {}
        for i, b in enumerate(self):
            if b.identifier not in state_actions:
                state_actions[b.identifier] = (self[i].L, self[i].A)
        return state_actions
    
    def randomise_states(self, p = None, steady_dist = False):
        for b in self:
            b.state = b.random_state(p = p, steady_dist = steady_dist)
    
    def random_actions(self):
        actions = list()
        for b in self:
            actions.append(b.random_action())
        return actions
    
    def get_bandits(self, identifiers, first = True):
        bandits = RMAB([b for b in self if b.identifier in identifiers])
        if first:
            return bandits[0]
        else:
            return bandits
    
    def get_wi(self, **kwargs):
        wi = {}
        for i, b in enumerate(self):
            if b.identifier not in wi:
                wi[b.identifier] = b.whittle_index(**kwargs)
        return wi
            

class Bandit:
    '''
    a template class of a Bandit.
    The constructor takes input L which is the number of states the bandit can
    occupy, A which is the number of actions, state which is the initial state,
    identifier which is a string that identifies this type of bandit.
    
    states are the set of integers in the interval [0,L-1]
    actions are the set of integers in the interval [0,A-1]
    
    properties
    L - the number of states
    A - the number of actions
    state - the current state of the bandit
    identifier - an identifier of the type of bandit
    
    methods
    transition
    input an action. a single time step is simulated based on the current state
    and inputted action. The reward is returned if return_reward is true.
    
    reward
    a reward given the previous state (state0), the action (action) and the next state (state1) is simulated and returned
    
    random_state
    returns a random state
    
    transition_prob
    calculates the probability of transitioning to each state given the current state and action (identity by default)
    
    random_action
    returns a random action
    '''
    def __init__(self, L, A = 2, state = None, identifier = None):
        self.L = L
        self.A = A
        self.bandit_args = dict()
        
        if identifier is None:
            self.identifier = 'Bandit'
        else:
            self.identifier = identifier
        
        if state is None:
            self.state = self.random_state()
        else:
            self.state = state
    
    def __repr__(self):
        return 'Bandit - ' + self.identifier
    
    def __str__(self):
        return self.identifier
    
    def transition(self, action, return_reward = True):
        state0 = self.state
        transition_prob = self.transition_prob(state0, action)
        self.state =  self.random_state(p = transition_prob)
        if return_reward:
            return self.reward(state0, action, self.state)
    
    def reward(self,state0, action, state1):
        # overwrite this method in the subclass
        # default reward is nill
        return 0.0
    
    def random_state(self, p = None, steady_dist = False):
        if steady_dist:
            p = self.steady_dist
        return np.random.choice(range(self.L), p = p)
    
    def transition_prob(self, state, action):
        # overwrite this method in the subclass
        # default transition is stationary
        identity = np.zeros(self.L)
        identity[state] = 1
        return identity
    
    def random_action(self, p = None):
        return np.random.choice(range(self.A), p = p)


class BanditPR(Bandit):
    '''
    Subclass of the Bandit template where the transition probabilities and
    expected rewards are known.
    '''
    def __init__(self, L, A = 2, identifier = None,
                 P = None, R = None, loadpath = None):
        if loadpath is not None:
            self.P, self.R = self.load_PR(path = loadpath)
            L = self.P.shape[-1]
        
        if L < 3:
            raise Exception('There must be at least 3 states')
        
        if P is not None:
            self.P = P
        if R is not None:
            self.R = R
        
        super().__init__(L, A = A, identifier = identifier)
        self.steady_dist = steady_state_matrix(self.P[0,:,:]).mean(0)
        self.discount_max = indexability_bound(self.P)
    
    def reward(self,state0, action, state1):
        return self.R[action,state0]
    
    def transition_prob(self, state, action):
        return self.P[action,state,:]
    
    def save_PR(self, path = ""):
        if len(path) > 0:
            if not os.path.exists(path):
                os.makedirs(path)
        np.savetxt(os.path.join(path,"R.csv"), self.R, delimiter = ",")
        np.savetxt(os.path.join(path,"P0.csv"),self.P[0,:,:], delimiter = ",")
        np.savetxt(os.path.join(path,"P1.csv"),self.P[1,:,:], delimiter = ",")
    
    def load_PR(self, path = ""):
        R = np.loadtxt(os.path.join(path,"R.csv"), delimiter = ",")
        P0= np.loadtxt(os.path.join(path,"P0.csv"), delimiter = ",")
        P1= np.loadtxt(os.path.join(path,"P1.csv"), delimiter = ",")
        P = np.stack([P0,P1],0)
        return P, R
    
    def whittle_index(self, **kwargs):
        return whittle_index(self.P, self.R, **kwargs)


class RandomBandit(BanditPR):
    '''
    A random bandit. For each state-action pair there are 'nonzero' number of
    non-zero transition probablities, such that the two stochastic matrices
    are both irreducible.
    '''
    def __init__(self, L = 4, nonzero = 3):
        self.nonzero = nonzero
        P0 = random_irreducible_P(L, nonzero = nonzero)
        P1 = random_irreducible_P(L, nonzero = nonzero)
        self.P = np.stack([P0,P1],0)
        self.R = np.zeros([2,L])
        self.R[1,:] = np.random.rand(L)
        super().__init__(L, A = 2, identifier = 'RandomBandit'
                         + str(np.random.randint(100000000,999999999)))
        self.bandit_args = {"nonzero" : nonzero}

class CycleBandit(BanditPR):
    '''
    The type of restless bandit used in Fu, et al. (2019)
    The bandit has a 50% of staying in the same state and a 50% chance of
    incrementing up when active or down when passive. The rewards are
    independent of the actions and are zero except in the first state and last
    state which yield -1 and 1 respectively.
    The constructor optional argument L is the number of states which is 4 by
    default (same as the paper).
    
    properties
    P - the transition matrices stacked into a 3d numpy array of size LxLx2
    R - the rewards of each state and action in a 2d numpy array of size Lx2
    (both columns are identical in this case)
    wi - an array of Whittle indices (with a discount factor of 1)
    
    methods
    permute_rows - helper function that permutes the rows by k steps of a 2d
    array M (returns new array)
    reward and transition_prob supersede the superclass methods
    '''
    def __init__(self, L = 4):
        
        identity = np.eye(L)
        P1 = 0.5*(identity + self.permute_rows(identity,1))
        P0 = self.permute_rows(P1,-1)
        self.P = np.stack([P0,P1],0)
        self.R = np.zeros([2,L])
        self.R[:, 0] = -1
        self.R[:,-1] =  1
        # Whittle Indices (when discount = 1)
        self.wi = np.array([(2.0*i-L+4)/L for i in range(L)])
        self.wi[0] = (2.0-L)/L
        self.wi[-1] = -1
        super().__init__(L, A = 2, identifier = 'CycleBandit' + str(L))
    
    def permute_rows(self, M, k):
        n = M.shape[0]
        return M[(np.arange(n)+k)%n,:]


class RestartBandit(BanditPR):
    '''Bandit from Avrachenkov and Borkar (2020) (B Example with restart).
    
    This bandit gradually progress their state with a low chance of resetting
    back to the first state when passive. They are reset to the first state
    with certaintity when active. The rewards decay exponentially with the
    state number when passive and are nill when active.
    
    L - int >= 3 - the number of states
    a - float in (0,1) - the base of the reward decay
    prestart - float in [0,1) - the probability of restarting when passive
    '''
    def __init__(self, L = 5, a = 0.9, prestart = 0.1):
        self.a = a
        self.prestart = prestart
        iden = 'RestartBandit_L='+str(L) + '_a='+str(a) + '_p='+str(prestart)
        P1 = np.zeros([L,L])
        P1[:,0] = 1.0
        P0 = np.zeros([L,L])
        P0[:,0] = prestart
        P0[:-1, 1:] = (1-prestart)*np.eye(L-1)
        P0[-1,-1] = (1-prestart)
        self.P = np.stack([P0,P1],0)
        self.R = np.zeros([2, L])
        self.R[0,:] = a**np.arange(1,L+1)
        super().__init__(L, A = 2, identifier = iden)
        self.bandit_args = {"a" : a, "prestart" : prestart}


class MentoringInstructions(BanditPR):
    ''' Mentoring Instructions example Bandit from Fu et al. (2019).
    
    The bandit represents a student and the states are the student's  study
    levels. At each state the student has a chance to either increase or
    decrease study level by 1 (except at the edges) with probabilities
    depending on if they are being mentored (active) or not (passive). The
    reward function is R(x) = sqrt(x/L) where L is the number of states (study
    levels) and x is the state (starting from 1).
    '''
    def __init__(self, L = 10, plearn0 = 0.3, plearn1 = 0.7):
        iden = 'MentoringInstructions_L='+str(L) + '_p0='+str(plearn0) + '_p1='+str(plearn1)
        self.plearn0 = plearn0
        self.plearn1 = plearn1
        P0 = np.zeros([L,L])
        P1 = np.zeros([L,L])
        ind0, ind1 = range(L-1), range(1,L)
        P0[ind0,ind1] = plearn0
        P0[ind1,ind0] = 1.0 - plearn0
        P0[0,0] = 1.0 - plearn0
        P0[-1,-1] = plearn0
        P1[ind0,ind1] = plearn1
        P1[ind1,ind0] = 1.0 - plearn1
        P1[0,0] = 1.0 - plearn1
        P1[-1,-1] = plearn1
        self.P = np.stack([P0,P1],0)
        self.R = np.tile(np.sqrt(np.arange(1,L+1)/L),[2,1])
        super().__init__(L, A = 2, identifier = iden)
        self.bandit_args = {"plearn0" : plearn0, "plearn1" : plearn1}










