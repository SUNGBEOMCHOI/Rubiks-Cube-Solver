import logging
import math

import numpy as np
import copy

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, model, cfg):

        """
        env : cube class
        model : trained neural network
        cfg : configuration(numMCTSSims, cpuct)
        """

        self.model = model
        self.numMCTSSim = cfg['mcts']['numMCTSSim']
        self.cpuct = cfg['mcts']['cpuct']
        self.action_dim = 6

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores env.getenvEnded ended for board s

    def getActionProb(self, state, env, depth, temp=0):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state.
        Args:
            state : numpy array which represent state  (7,21)
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)   , list type
        """
        self.initial_env = env
        self.env = copy.deepcopy(env)

        # numMCTSSims : number of MCTS simulations
        for i in range(self.numMCTSSim):
            self.search(state, depth)
            self.env = copy.deepcopy(self.initial_env)

        s = np.array2string(state)
        counts = [self.Nsa[(depth, s, a)] if (depth, s, a) in self.Nsa else 0 for a in range(self.action_dim)] #(6,) list  number of times that (s,a) was visited

        if temp == 0:
            #greedy
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        # elif temp == -1:
        #     a = [self.Qsa[(depth, s, a)] if (depth, s, a) in self.Qsa else 0 for a in range(self.action_dim)]
        #     bestA = a.index(max(a))
        #     probs = [0]*6
        #     probs[bestA] = 1
        #     return probs


        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state, depth=0, previous_action = -1, env_ended = False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Args:
            state : numpy array which represent state  (7,21)
            depth : depth of node
            previous_action : integer(0-action_dim) which represent previous action
            env_ended : Boolean type
        Returns:
            v: the negative of the value of the current state (if state is terminal, return 1)
        """

        s = np.array2string(state)
        if previous_action == -1:
            invalid_action = -1
        else:
            invalid_action = previous_action + 1 - 2*(previous_action%2)

        if (depth, s) not in self.Es:
            self.Es[(depth, s)] = env_ended
    

        if self.Es[(depth, s)]:
            # terminal node is found
            return 1


        if (depth, s) not in self.Ps:
            # leaf node is found

            v, self.Ps[(depth, s)] = self.model.predict(state)
            if invalid_action != -1:
                self.Ps[(depth, s)][invalid_action] = 0
                sum_Ps_s = np.sum(self.Ps[(depth,s)])
                self.Ps[(depth, s)] /= sum_Ps_s

            self.Ns[(depth, s)] = 0
            return v


        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.action_dim):
            if a != invalid_action:
                if (depth, s, a) in self.Qsa:
                    u = self.Qsa[(depth, s, a)] + self.cpuct * self.Ps[(depth, s)][a] * math.sqrt(self.Ns[(depth, s)]) / (
                            1 + self.Nsa[(depth, s, a)])
                else:
                    u = self.cpuct * self.Ps[(depth, s)][a] * math.sqrt(self.Ns[(depth, s)])  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, _, done, _ = self.env.step(a)


        v = self.search(next_s, depth+1, a, env_ended=done)

        

        if (depth, s, a) in self.Qsa:
            self.Qsa[(depth, s, a)] = max(self.Qsa[(depth, s, a)], v)
            self.Nsa[(depth, s, a)] += 1

        else:
            self.Qsa[(depth, s, a)] = v
            self.Nsa[(depth, s, a)] = 1

        self.Ns[(depth, s)] += 1
        return v


    def visualization(self):
        encoded_state = {}
        depth_list = []
        for k in range(70):
            depth_list.append([])
        

        for idx, key in enumerate(self.Ps.keys()):
            if key[1] not in encoded_state:
                encoded_state[key[1]] = idx
            
        
        for key in self.Ps.keys():
            depth_list[key[0]].append(encoded_state[key[1]])
        
        print(depth_list)