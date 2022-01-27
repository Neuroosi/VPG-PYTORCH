import numpy as np

class Transition:
    def __init__(self, actionspacesize):
        self.actionSpaceSize = actionspacesize
        self.states = []
        self.rewards = []
        self.actions = []
        self.probs = []

    def addTransition(self, state, reward, action):
        self.states.append(state)
        self.rewards.append(reward)
        cache = np.zeros(self.actionSpaceSize)
        cache[action] = 1
        self.actions.append(cache)

    def resetTransitions(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.probs = []

    def discounted_reward(self, GAMMA):
        G = np.zeros(len(self.rewards))
        ##Calculate discounted reward
        cache = 0
        for t in reversed(range(0, len(self.rewards))):
            if self.rewards[t] != 0: cache = 0
            cache = cache*GAMMA + self.rewards[t]
            G[t] = cache
        ##Normalize
        G = (G-np.mean(G))/(np.std(G))
        return G