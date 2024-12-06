import numpy as np
import matplotlib.pyplot as plt
from warnings import warn


class Collection:
    """Super class for lists or dictionaries to run methods or get attributes.
    
    Methods
    -------
    run : executes a common method in every element of the collection.
    get : returns a list or dictionary of a common attribute.
    """
    
    def run(self, func, *args, **kwargs):
        """Executes a common method in every element of the collection.
        
        The method denoted by func is executed for each element in the
        collection using *args and **kwargs as arguments.

        Parameters
        ----------
        func : str
            The name of the method to be executed as a string.
        *args : tuple
            Arguments to be included in the method.
        **kwargs : dictionary
            Keyword arguments to be included in the method.

        Returns
        -------
        None.

        """
        if isinstance(self, dict):
            for e in self:
                getattr(self[e], func)(*args, **kwargs)
        else:
            for e in self:
                getattr(e, func)(*args, **kwargs)
    
    def get(self, attr):
        """Returns a list or dictionary of a common attribute.
        
        The attribute denoted by attr is extracted from every element of the
        collection and returned as either a list or dictionary.
        
        
        Parameters
        ----------
        attr : str
            The name of the common attribute as a string.

        Returns
        -------
        list/dict
            List or dictionary of the attribute from each element of the
            collection.

        """
        if isinstance(self, dict):
            return {e : getattr(self[e], attr) for e in self}
        else:
            return [getattr(e, attr) for e in self]



def check_stochasticity(P, tol = 1e-14):
    """Identifies stochastic dimensions of the array P.
    
    A dimension is considered stochastic when every element of P is within the
    interval [0-tol,1+tol] and summing across that dimension gives a constant
    array of ones (each sum is within [1-tol,1+tol]).

    Parameters
    ----------
    P : float array
        An array of any size.
    tol : float >= 0, optional
        Tolerance. Absolute values less than tol are considered as zero.
        The default is 1e-14.

    Returns
    -------
    stochastic_dims : list
        A list of dimensions that are stochastic. If no dimensions are
        stochastic then an empty list is returned.

    """
    
    stochastic_dims = list()
    
    # check that every element is in the interval [0-tol,1+tol]
    if (P < 0-tol).any() or (P > 1+tol).any():
        return stochastic_dims
    
    # append dimensions that sum to 1 within the tolerance
    for dim in range(P.ndim):
        if (abs(P.sum(dim)-1)<=tol).all():
            stochastic_dims.append(dim)
    return stochastic_dims


def steady_state_matrix(P, check = True, tol = 1e-14):
    """Calculates the steady state array of P (Cesaro-limit).
    
    "COMPUTATION OF THE EIGENPROJECTION OF A NONNEGATIVE MATRIX AT ITS
    SPECTRAL RADIUS" by Uriel G. Rothblum,
    Mathematical Programming Study 6 (1976) 188-201.
    
    Parameters
    ----------
    P : (..., L, L) float array
        Stacked stochastic matrices with L states.
    check : boolean, optional
        Whether or not to check stochasticity. The default is True.
    tol : float > 0, optional
        Threshold to consider a value as zero. The default is 1e-14.

    Raises
    ------
    Exception
        The last dimension of P must be stochastic. Checked if check is True.

    Returns
    -------
    (..., L, L) float array
        Stochastic array of the same shape as P. The Cesaro-limit.

    """
    
    if check:
        # the last two dims of P should be stacked right stochastic matrices
        if (P.ndim-1) not in check_stochasticity(P):
            raise Exception('The last dimension of P is not stochastic')
    
    ogshape = P.shape
    P2 = P.reshape(-1,ogshape[-2],ogshape[-1])
    Pinf = np.zeros(P2.shape)
    for i in range(P2.shape[0]):
        p = P2[i,:,:]
        w1,v1 = np.linalg.eig(p)
        w2,v2 = np.linalg.eig(p.T)
        X = v1[:,abs(w1-1)<tol]
        Y = v2[:,abs(w2-1)<tol].T.conjugate()
        Pinf[i,:,:] = np.real(X@np.linalg.solve(Y@X, Y))
    
    return Pinf.reshape(ogshape)


def steady_state_matrix_sum(P, check = True, tol = 1e-14, max_it = 1e4):
    '''
    Calculates the steady state array of P (Cesaro-limit) using a summation.
    P should be a stacked array of right stochastic matrices in the last two
    dimensions. The stacked array of the same size containing the steady state
    matrices in the last two dimensions is returned. The steady state matrix
    is defined by:
    S = \lim_{T \to \infty}\frac{1}{T}\sum_{i = 1}^T P^i,
    which results in a right stochastic matrix satisfying S = SP = PS.
    If \lim_{T \to \infty}P^T converges then it also converges to S. However,
    the sum becomes necessary when that limit does not converge due to
    periodic behaviour.
    
    'check' is a boolean which decides if the stochasticity of P needs to be
    checked. 'tol' is the tolerance used when checking convergence. 'max_it'
    is the maximum number of iterations allowed.
    '''
    if check:
        # the last two dims of P should be stacked right stochastic matrices
        if (P.ndim-1) not in check_stochasticity(P):
            raise Exception('The last dimension of P is not stochastic')
    
    L = P.shape[-1]
    # start the sum at a high power to accelerate convergence
    exponent = 1000000*(L-1)
    P0 = np.linalg.matrix_power(P, exponent)
    T = 1
    Pinf = P0.copy()
    error = abs(P@Pinf - Pinf).max()
    # average further powers until convergence
    while error > tol:
        if T >= max_it:
            warn('Reached maximum iterations before convergence with error'\
                 + '{}'.format(error))
            break
        T += 1
        exponent += 1
        if exponent % 10 == 0:
            P0 = np.linalg.matrix_power(P, exponent)
        else:
            P0 = P@P0
        Pinf += (P0 - Pinf)/T
        error = abs(P@Pinf - Pinf).max()
    return Pinf



def greedy_policy(Q = None, V = None, P = None, R = None,
                  discount = 1, epsilon = 0):
    '''
    Returns an array of actions for each state of size (L,), representing the
    greedy policy. 'epsilon' is the probability of choosing random actions
    representing an epsilon greedy policy (all actions in the array are
    random or none of them). The greedy policy is either calculated from the
    Q-function 'Q' or the value function 'V'. Exactly one of these functions
    must be provided in the form of an array of size (A,L) or (L,)
    respectively. If V is provided then the transition array 'P' of size
    (A,L,L) and the expected reward array 'R' of size (A,L) must also be
    given. 'discount' should be a scaler in the interval [0,1] representing
    the discount factor.
    '''
    
    if (Q is None) == (V is None):
        raise Exception('Exactly one of Q and V should be provided')
    
    # random policy with probability epsilon
    if epsilon > 0:
        if np.random.rand() < epsilon:
            if V is None:
                A = Q.shape[0]
                L = Q.shape[1]
            else:
                A = P.shape[0]
                L = P.shape[-1]
            policy = np.random.randint(0, A, L)
            return policy
    
    # calculate Q from V
    if V is not None:
        Q = R + discount*P@V
    # greedy policy chooses the action which maximises Q
    policy = Q.argmax(0)
    return policy


def value_iteration(P, R, discount = 1, relative = True,
                    thresh_V = 1e-14, max_it = 1e4, check = True,
                    V = None):
    '''
    Perform value iteration to estimate the value function of an MDP.
    P should be an array of transition matrices of shape (A,L,L) where A is
    the number of actions and L is the number of states. R is an array of
    expected rewards at each state for each action of size (A,L). 'discount'
    is the discount factor. 'relative' is a boolean indicating if the relative
    reward should be calculated. The algorithm might not converge if ralative
    is False and discount is 1. 'thresh_V' is the maximum change in V to
    establish convergence. 'max_it' is the maximum allowed number of
    iterations. A warning will appear if the algorthim terminates from this
    condition. 'check' is a boolean to check if P is stochastic. 'V' is an
    optional initial estimate for the value function.
    An array of values (V) for each state of shape (L,) is returned.
    '''
    
    # undiscounted non-relative value function is not unique
    if discount == 1 and not relative:
        warn('attempting to calculate undiscounted non-relative values')
    
    if check:
        # the last two dims of P should be stacked right stochastic matrices
        if (P.ndim-1) not in check_stochasticity(P):
            raise Exception('The last dimension of P is not stochastic')
    
    L = P.shape[-1]
    if V is None:
        # intialise with V = 0 and g = 0
        V0 = np.zeros(L)
        V = R.max(0)
    else:
        V0 = V-1
    count = 0
    error = abs(V-V0).max()
    while error > thresh_V:
        if count >= max_it:
            warn('Reached maximum iterations before convergence with error'\
                 + '{}'.format(error))
            break
        count += 1
        V0 = V
        Q = R + discount*P@V
        # calculate the maximum long term average reward when relative is True
        if relative:
            policy = Q.argmax(0)
            Ppolicy = P[policy,range(L),:]
            Rpolicy = R[policy,range(L)]
            Pinf = steady_state_matrix(Ppolicy, check = False)
            g = Pinf@Rpolicy # long term average reward
            V = Q.max(0)-g.max()
        else:
            V = Q.max(0)
        error = abs(V-V0).max()
    print(g)
    return V
    

def value_inversion(P, R, discount = 1, relative = True):
    """Calculate the values of a Markov Reward processes
    
    The values are defined by V = R + discount*P@V which are solved by
    V = (I-discount*P)^-1@R when relative is False, or the relative values
    V = (I-Pinf)@R + discount*P@V are solved by
    V = (I-discount*(P-Pinf))^-1@(I-Pinf)@R when relative is True.
    Where Pinf the Cesaro limit of P.

    Parameters
    ----------
    P : (...,L,L) float array
        Stacked transition matrices with L states.
    R : (...,L) float array
        Expected rewards at each state.
    discount : float in [0,1], optional
        Discount factor. The default is 1.
    relative : boolean, optional
        Compute relative values if True. The default is True.

    Raises
    ------
    Exception
        relative must be True if discount is one.

    Returns
    -------
    V : (...,L) float array
        (Relative) Values at each state.
    g : (...,L) float array
        Long term average reward when starting at each state. For systems with
        only a single communicating class the last dimension is constant.

    """
    
    if discount == 1 and not relative:
        raise Exception('cannot calculate undiscounted non-relative values')
    
    L = P.shape[-1]
    I = np.eye(L)
    for i in range(P.ndim - 2):
        I = np.expand_dims(I, 0)
    R = np.expand_dims(R,-1)
    Pinf = steady_state_matrix(P, check = False)
    g = Pinf@R # long term average reward
    if relative:
        V = np.linalg.solve(I - discount*(P-Pinf), R - g)
    else:
        V = np.linalg.solve(I - discount*P, R)
    V = np.squeeze(V, axis = -1)
    g = np.squeeze(g, axis = -1)
    return V, g
    

def policy_iteration(P, R, discount = 1, relative = True,
                     max_it = 1e3, check = True, policy = None,
                     alpha = 0.001):
    '''
    Use the policy iteration algorthm to find a deterministic optimal policy
    and corresponding value function. P should be an array of transition
    matrices of shape (A,L,L) where A is the number of actions and L is the
    number of states. R is an array of expected rewards at each state for each
    action of size (A,L). 'discount' is the discount factor. 'relative' is a
    boolean indicating if the relative reward should be calculated. 'check' is
    a boolean to check if P is stochastic. 'policy' is an optional initial
    estimate for the policy. 'max_it' is the maximum number of iterations.
    The function returns the tuple (policy, V) which is the optimal
    deterministic policy and corresponding value function is array forms of
    size (L,).
    '''
    
    # undiscounted non-relative value function is not unique
    if discount == 1 and not relative:
        warn('attempting to calculate undiscounted non-relative values')
    
    if check:
        # the last two dims of P should be stacked right stochastic matrices
        if (P.ndim-1) not in check_stochasticity(P):
            raise Exception('The last dimension of P is not stochastic')
    
    L = P.shape[-1]
    if policy is None:
        policy = np.zeros(L, dtype = int)
    policy0 = 1 - policy
    policy_previous = tuple(policy0)
    count = 0
    policy_tried = set()
    while tuple(policy) != policy_previous:
        if count >= max_it:
            warn('Reached maximum iterations before convergence')
            break
        if tuple(policy) in policy_tried:
            warn('Policy iteration halted after reaching cycle')
            break
        count += 1
        policy0 = policy.copy()
        policy_previous = tuple(policy0)
        policy_tried.add(policy_previous)
        Ppolicy = P[policy,range(L),:]
        Rpolicy = R[policy,range(L)]
        # evaluate value given the policy
        V, g = value_inversion(Ppolicy, Rpolicy, discount = discount,
                               relative = relative)
        # update the policy greedily
        if relative:
            bias = R - g + discount*P@V
            gain = P@g
            policy = (alpha*bias + (1-alpha)*gain).argmax(0)
        else:
            policy = (R + discount*P@V).argmax(0)
    
    return policy, V, g

def whittle_index_pi(P, R, discount = 1, relative = True, wi = None, lr = 0.1,
                     thresh_lam = 1e-13, max_it = 1e3):
    '''
    Calculates the Whittle Indices as an array for each state.
    'P' is an array of transition matrices of size (A,L,L)
    'R' is an array of rewards of size (A,L)
    'wi' is an array of optional initial guesses of the indices
    'lr' is the learning rate of the indices a scalar in the inverval (0,1]
    'thresh_lam' is the threshold change in index to decide convergence
    The function returns an array of Whittle indices of size (L,)
    
    The algorithm works as follows
    for each state
        initialise Whittle index estimate (wi0)
        do until convergence or reach iteration limit
            calculate the value function (V) using policy iteration
            calculate the the expected Whittle index (wi1) from V
            update the index estimate as the weighted average of wi0 and wi1
            error = abs(wi1-wi0)
            w0 = w1
    '''
    L = P.shape[-1]
    A1 = np.zeros([2,L])
    A1[1,:] = 1
    DR = R[1,:] - R[0,:]
    DP = P[1,:,:] - P[0,:,:]
    if wi is None:
        wi = np.zeros(L)
    for s in range(L):
        wi0 = wi[s]-1
        error = abs(wi[s]-wi0)
        policy = np.zeros(L, dtype = int)
        count = 0
        while error > thresh_lam:
            if count >= max_it:
                warn('Reached maximum iterations before convergence of '\
                     +'Whittle Index {} with error {}'.format(s,error))
                break
            count += 1
            wi0 = wi[s]
            Rwi = R - wi0*A1
            policy, V, g = policy_iteration(
                P, Rwi, discount = discount, relative = relative,
                check = False, policy = policy)
            wi1 = DR[s] + discount*DP[s,:]@V
            wi[s] = wi0 + lr*(wi1 - wi0)
            error = abs(wi1 - wi0)
        print(s, wi[s], count, error)
    return wi


def random_stochastic_array(*shape):
    """Generates a random stochastic array
    
    Generates a random array of values in the range [0,1] and then normalises
    the last dimension so that the sums across the last dimension are all 1.
    The shape is given by *shape. If the value of any sum is less than 1e-14
    then the function calls itself to generate a new array. 

    Parameters
    ----------
    *shape : ints
        The size of each dimension of the array to be generated.

    Returns
    -------
    P : (*shape) float array
        A random array that is stochastic in the last dimension.

    """
    
    P = np.random.rand(*shape)
    Psum = P.sum(-1)
    if (Psum < 1e-14).any():
        P = random_stochastic_array(*shape)
    else:
        P = P/np.expand_dims(Psum,-1)
    return P


def random_irreducible_P(L, nonzero = 3, max_attempt = 1000,
                         reducible = False):
    """Generates a random irreducible stochastic matrix
    
    The matrix is generated with fixed number of non-zero entries in each row.

    Parameters
    ----------
    L : int > 1
        Number of states.
    nonzero : int > 1, optional
        Number of non-zero transition probabilities. The default is 3.
    max_attempt : int > 0, optional
        Maximum number of attempts to generate matrix. The default is 1000.

    Raises
    ------
    Exception
        If the matrix is not succesfully generated in max_attempt attempts
        then an error will be given.

    Returns
    -------
    (L,L) float array
        Irreducible stochastic matrix.

    """
    
    p = random_stochastic_array(L, nonzero)
    P = np.zeros([L, L])
    for l in range(L):
        ind = np.random.choice(L, nonzero, replace = False)
        P[l,ind] = p[l,:]
    if isirreducible(P) != reducible:
        return P
    else:
        if max_attempt > 1:
            return random_irreducible_P(L, nonzero = nonzero,
                                        max_attempt = max_attempt - 1,
                                        reducible = reducible)
        else:
            raise Exception('maximum counts reached')


def whittle_indexOLD(P, R, discount = 1, relative = True):
    '''
    Returns an array of Whittle indices for each state of size (L,), where
    L is  the number of states.
    'P' is a stacked array of transition matrices of size (2,L,L), where the
    first index is the action (0 is passive and 1 is active)
    'R' is a stacked array of expected rewards at each state of size (2,L)
    'discout' is the discount factor
    'relative' is a boolean indicating if the value function is relative
    '''
    if discount == 1 and not relative:
        raise Exception('cannot calculate undiscounted non-relative values')
    
    L = P.shape[-1]
    I = np.eye(L)
    wi = np.full(L,np.inf)
    states_found = set()
    policy = np.zeros(L, dtype = int)
    DR = R[1,:] - R[0,:]
    DP = P[1,:,:] - P[0,:,:]
    while len(states_found) < L:
        Ppolicy = P[policy,range(L),:]
        Rpolicy = R[policy,range(L)]
        if relative:
            Pinf = steady_state_matrix(Ppolicy, check = False)
            M = discount*DP@np.linalg.solve(I-discount*(Ppolicy-Pinf), I-Pinf)
        else:
            M = discount*DP@np.linalg.solve(I-discount*Ppolicy, I)
        lam_pos = (DR + M@Rpolicy)/(1 + M@policy)
        #wimin = wi.min()
        #lam_new = max(l for i, l in enumerate(lam_pos) if l <= wimin and i not in states_found)
        lam_new = max(l for i, l in enumerate(lam_pos) if i not in states_found)
        #print(lam_pos)
        for i, l in enumerate(lam_pos):
            if l == lam_new and i not in states_found:
                wi[i] = lam_new
                states_found.add(i)
                policy[i] = 1
    return wi


def whittle_index(P, R, **kwargs):
    """Computes the Whittle indices from the transition probs and rewards.
    
    The Whittle indices can be computed from the transition probabilities and
    expected rewards using an algorithm similar to the "adaptive greedy"
    algorithm. Here an iterative cycle finds the bounds on the tax (lambda) in
    which a policy is optimal, and then updates the policy based on the state
    that limits the lower bound. This algorithm assumes that the optimal
    policy of a bandit for a given tax is unique, except at the Whittle
    indices where the optimal policies transition.

    Parameters
    ----------
    P : (2,L,L) float array
        Stacked array of transition matrices, where the first index is the
        action (0 is passive and 1 is active).
    R : (2,L) float array
        Stacked array of expected rewards at each state.
    **kwargs : keyword arguments
        Optional arguments for lambda_interval, such as the discount factor.

    Returns
    -------
    wi : (L,) float array
        Whittle index of each state.

    """
    
    L = P.shape[-1]
    policy = np.full(L, 0)
    # initial policy is a passive policy
    wi = np.zeros(L)
    identified = []
    prev_ind = -1
    while (policy == 0).any():
        # identify range of taxes in which policy is optimal
        d, ind, optimal = lambda_interval(policy, P, R, **kwargs)
        
        ind_new = [i for i in ind[0] if i not in identified]
        
        # check if index as already been identified
        if len(ind_new) > 0:
            identified.append(ind_new[0])
            # the Whittle index is the lower bound
            wi[ind_new[0]] = d[0]
            # policy is updated so that the state bounding the tax is active
            policy[ind_new[0]] = 1
            prev_ind = ind_new[0]
        else:
            # remove previous index to avoid cycles
            ind2 = [i for i in ind[0] if i != prev_ind]
            # policy is updated so that the state bounding the tax is switched
            policy[ind2[0]] = 1 - policy[ind2[0]]
            prev_ind = ind2[0]
    return wi




def policy_structure(P, R, policy = None, result = None, **kwargs):
    """Computes the tax (lambda) intervals for each optimal policy
    
    The optimal policy of a single restless bandit depdends on the tax
    (lambda) on the active action. There exists a threshold tax in which
    the passive policy is optimal for any tax larger than the threshold.
    Similarly, there is another threshold tax in which the active policy is
    optimal for any tax less than the threshold. Between these two thresholds
    the optimal policy changes at specific tax values. This function
    identifies all the optimal policies and associated intervals (inclusive)
    of tax in which the policy is optimal.

    Parameters
    ----------
    P : (2,L,L) float array
        Stacked array of transition matrices, where the first index is the
        action (0 is passive and 1 is active).
    R : (2,L) float array
        Stacked array of expected rewards at each state.
    policy : (L,) array, optional
        Starting policy, which should be an optimal policy.
        The default is a passive policy.
    result : dictionary, optional
        Initialisation of the result dictionary.
        The default is an empty dictionary.
    **kwargs : keyword arguments
        Optional arguments for lambda_interval, such as the discount factor.

    Returns
    -------
    result : dictionary
        A dictionary with policies as keys and lambda_interval outputs as
        values.

    """
    
    if policy is None:
        policy = np.zeros(P.shape[-1], dtype = int)
    
    if result is None:
        result = {}
    
    d, ind, optimal = lambda_interval(policy, P, R, **kwargs)
    if optimal:
        result[tuple(policy)] = (d, ind, optimal)
        for direction in [0,1]:
            for i in ind[direction]:
                policyi = policy.copy()
                policyi[i] = 1 - direction
                if tuple(policyi) not in result:
                    resulti = policy_structure(
                        P, R, policy = policyi, result = result, **kwargs)
                    result = {**result, **resulti}
    return result


def lambda_interval(policy, P, R, discount = 1, relative = True,
                    explore = [0, 0.5]):
    """Identifies the tax interval in which the policy is optimal
    
    For a policy to control a single restless bandit to be optimal the
    quality of each action dictated by the policy must be weakly greater than
    the quality of the alternative actions. This requirement gives a system
    of inequalities which constrain the tax values in which the policy can be
    optimal. This function computes the lower and upper bounds on the tax in
    which the policy is optimal as well as which states lie on the bounds. If
    a policy is never optimal then either the system of inequalities has no
    solution, or if relative is True and the transition probabilities are
    reducible under the policy, then a spurious interval could be returned.

    Parameters
    ----------
    policy : (L,) int array
        The policy as a vector of ones and zeros representing active and
        passive actions for each state.
    P : (2,L,L) float array
        Stacked array of transition matrices, where the first index is the
        action (0 is passive and 1 is active).
    R : (2,L) float array
        Stacked array of expected rewards at each state.
    discount : float/int, optional
        Discount factor, should be in the interval [0,1]. The default is 1.
    relative : Boolean, optional
        Use relative rewards if True. The default is True.
    explore : list, optional
        Two element list where the first element is the probability of
        exploration and the second element is the probability of activation
        when exploring. These values are used to modify the Whittle indices
        to account for an index policy with exploration. The default is
        [0, 0.5] indicating the exploration rate is 0.

    Returns
    -------
    d : tuple of floats
        The lower and upper bounds of lambda in which the policy is optimal.
    ind : tuple of lists of states
        The states which lie on the bounds.
    optimal : boolean
        False if the Q function suggests the policy is not optimal for any
        lambda. Note that in the relative case it can return True when there
        are multiple communicating classes, even when the policy is not
        optimal.

    """
    
    L = P.shape[-1]
    DR = R[1,:] - R[0,:]
    DP = P[1,:,:] - P[0,:,:]
    I = np.eye(L)
    
    Ppolicy = P[policy,range(L),:]
    Rpolicy = R[policy,range(L)]
    if explore[0]>0:
        Pexplore = explore[1]*P[1,:,:] + (1-explore[1])*P[0,:,:]
        Rexplore = explore[1]*R[1,:] + (1-explore[1])*R[0,:]
        Ppolicy = explore[0]*Pexplore + (1-explore[0])*Ppolicy
        Rpolicy = explore[0]*Rexplore + (1-explore[0])*Rpolicy
        policyexplore = explore[0]*explore[1] + (1-explore[0])*policy
    else:
        policyexplore = policy
    if relative:
        Pinf = steady_state_matrix(Ppolicy, check = False)
        M = discount*DP@np.linalg.solve(I-discount*(Ppolicy-Pinf), I-Pinf)
    else:
        M = discount*DP@np.linalg.solve(I-discount*Ppolicy, I)
    DQ = DR + M@Rpolicy
    DH = 1  + M@policyexplore
    cand = np.divide(DQ, DH, out = DQ, where = DH!=0)
    indp = ((policy==1) & (DH>0)) | ((policy==0) & (DH<0))
    indm = ((policy==1) & (DH<0)) | ((policy==0) & (DH>0))
    indn = DH == 0
    d = (cand[indm].max(initial = np.NINF), cand[indp].min(initial = np.inf))
    ind = (
        [i for i in range(L) if indm[i] and cand[i]==d[0]],
        [i for i in range(L) if indp[i] and cand[i]==d[1]]
        )
    # upper bound cannot be less than the lower bound
    optimal = (d[1] >= d[0])
    if np.any(indn):
        # DQ must be consistent with the policy when DH == 0
        optimal = optimal and np.all((DQ[indn] > 0)==(policy[indn]==1))
    return d, ind, optimal
    



def policy_evaluation(policy, P, R, discount = 1, relative = True):
    '''
    Evalutes the values and long term expected reward rate of a given
    deterministic policy.
    'policy' is an array of actions of length L
    'P' is an array of stacked transition matrices of size (A,L,L)
    'R' is an array of rewards of size (A,L)
    'discount' is the discount factor
    'relative' is a boolean indicating if the relative values are computed
    A tuple of arrays of length L are returned containing the (relative)
    values and long term expected reward rates at each state.
    '''
    L = P.shape[-1]
    Pp = P[policy,range(L),:]
    Rp = R[policy,range(L)]
    return value_inversion(Pp, Rp, discount = discount, relative = relative)


def resolvent_update_series(alpha, discount, res0, U_P,
                            max_order = 1, thresh = None):
    '''
    Given the update from stochastic matrix P to P+alpha*(U-P) where U is also
    a stochastic matrix, then this function approximates the new resolvent
    (I-discount*(P+alpha*(U-P)))^-1 from the previous resolvent
    (I-discount*P)^-1 using the Taylor series about alpha = 0.
    
    alpha - float in [0,1] - is the weight of U in the update
    discount - float in [0,1) - is the discount factor
    res0 - float array (L,L) - is the previous resolvent (from P)
    U_P - float array (L,L) - U subtract P
    max_order - int >= 0 - the maximum order of the series approximation
    thresh - float > 0 - threshold for term magnitude to continue the series
    '''
    term = res0.copy()
    res1 = term
    for i in range(1, max_order+1):
        term = alpha*discount*res0@U_P@term
        res1 += term
        if thresh is not None:
            if abs(term).max() <= thresh:
                break
    return res1

def resolvent(P, discount):
    '''
    Calculates the resolvent of the stochastic matrix P: (I-discount*P)^-1
    
    P - stochastic array (L,L) - transition matrix
    discount - float in [0,1) - is the discount factor
    '''
    I = np.eye(P.shape[-1])
    return np.linalg.solve(I-discount*P, I)

def stochastic_pert_decom(P1, P2, alpha = None):
    '''
    Decomposes the stochastic matrix P2 as a purturbation of the stochastic
    matrix P1 of the form P2 = P1 + alpha*(U - P1), where alpha is a scaler
    in the interval [0,1] and U is a stochastic matrix. If alpha is not
    specified then the function computes the smallest value of alpha that
    permits U to remain between 0 and 1. The function returns U and alpha.
    
    Args
    P1 - float array (L,L) - the stochastic matrix being perturbed
    P2 - float array (L,L) - the resulting stochastic matrix
    alpha - float in (0,1] - weight of U in the weighted average
    
    Returns
    U - float array (L,L) - perturbation stochastic matrix
    alpha - float in [0,1] - weight of U in the weighted average
    '''
    if alpha is None:
        alpha = max([np.nanmax(((P1-P2)/P1)),np.nanmax(((P2-P1)/(1-P1)))])
    U = (P2-(1-alpha)*P1)/alpha
    return U, alpha

def isirreducible(P, thresh = 1e-14, returng = False):
    '''
    Checks if stochastic matrices are irreducible.

    Parameters
    ----------
    P : (..., L, L) float array
        Stacked stochastic matrices where L is the number of states.
    thresh : float >= 0, optional
        Threshold for probabilities to be considered zero.
        The default is 1e-14.
    returng : boolean, optional
        Set to True to return the g, the connection array.
        The default is False.

    Returns
    -------
    (...) boolean array
        True if corresponding schocastic matrix is irreducible.
    (..., L, L) binary array
        Returns if returng is True.
        1 if state can transition eventually, 0 otherwise.

    '''
    n = P.ndim
    g0 = 1*(P > thresh)
    g = (g0 + g0@g0).clip(max = 1)
    while not np.array_equal(g0, g):
        g0 = g
        g = (g0 + g0@g0).clip(max = 1)
    isir = np.all(g == 1, axis = (n-2,n-1))
    if returng:
        return isir, g
    else:
        return isir

def indexability_bound(P, b0 = 1.0, thresh = 1e-14):
    ''' Returns an upper bound to the discount factor that is a sufficient
    condition for indexability. Viz. if the discount factor is <= the returned
    value then the system is indexable according to proposition 2 in
    "Restless bandits: indexability and compuation of Whittle index" by
    Akbarzadeh and Mahajan 2020.

    Parameters
    ----------
    P : (2,L,L) float array
        The transition matrices.
    b0 : float, optional
        Intial guess of discount factor. The default is 1.0.
    thresh : float, optional
        Maximum difference between inequality sides. The default is 1e-14.

    Returns
    -------
    float
        The largest (within tolerance) discount factor that satisfies the
        indexability sufficient condition.

    '''
    # find the maximum discount factor that satisifies the first condition
    # iteratively
    P1 = P[1,:,:]
    L = P.shape[-1]
    Pz = np.repeat(P1, L, axis = 0)
    Px = np.tile(P1, [L,1])
    b1 = 1 - b0
    LHS, RHS = 0, 1
    while (abs(LHS-RHS) >= thresh) or (LHS > RHS):
        b1 = b0
        LHS = (b1*Pz-Px).clip(min = 0.0).sum(-1).max()
        RHS = (1-b1)**2/b1
        b0 = 0.5*(2+LHS-np.sqrt(LHS*(4+LHS)))
    
    # find the maximum discount factor that satisifies the second condition
    b2 = 1/(1+(P[0,:,:]-P[1,:,:]).clip(min = 0.0).sum(-1).max())
    return max([0.5,b1,b2])
    


def whittle_dependence(P,R, vkwarg, **kwargs):
    """Compute the Whittle indices for a range of parameters
    
    For example:
    whittle_dependence(P, R, {"discount" : [0.5,0.9,0.99]}, relative = True)
    will compute the Whittle indices using the discount values of 0.5, 0.9 and
    0.99, all while keeping relative as True.

    Parameters
    ----------
    P : (2,L,L) float array
        Stacked array of transition matrices, where the first index is the
        action (0 is passive and 1 is active).
    R : (2,L) float array
        Stacked array of expected rewards at each state.
    vkwarg : dictionary
        Dictionary containing a single key with a list of values. The
        key corresponds to an optional argument for lambda_interval and the
        values represent the range of values to compute the Whittle indices.
    **kwargs : keyword arguments
        Optional arguments for lambda_interval, such as the discount factor.

    Returns
    -------
    wi : (n,L) float array
        Stacked array of Whittle indices where the first dimension indexes
        based on the vkwarg values.

    """
    
    L = P.shape[-1]
    vals = list(vkwarg.values())[0]
    var = list(vkwarg.keys())[0]
    n = len(vals)
    wi = np.zeros([n,L])
    for i, v in enumerate(vals):
        wi[i,:] = whittle_index(P, R, **{var : v}, **kwargs)
    return wi



