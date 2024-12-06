


def WIsim(rmab, controller, T, episode, wiepisode):
    '''Simulates a Whittle index controller acting on an RMAB
    
    rmab - RMAB - the restless multiarmed bandit
    controller - QWIC - the controller that determins the policy
    T - int - time steps to simulate
    episode - WhittleEpisode - stores the episode information
    '''
    
    DT = T - episode.T + episode.t
    if DT > 0:
        episode.add_time(DT)
    
    states1 = rmab.states()
    wiepisode.step(controller)
    for t in range(episode.t, episode.t + T):
        states0 = states1
        actions = controller.policy(states0)
        rewards = rmab.transition(actions)
        states1 = rmab.states()
        episode.step(states1, rewards, actions)
        controller.update(episode)
        wiepisode.step(controller)


