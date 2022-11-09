import logging
import math
import random
from collections import deque

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
        cfg : configuration(numMCTSSims, cpuct, virtual_loss_const)
        """

        self.model = model
        self.numMCTSSim = cfg['mcts']['numMCTSSim']
        self.cpuct = cfg['mcts']['cpuct']
        self.virtual_loss_const = cfg['mcts']['virtual_loss_const']
        self.action_dim = 6

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Lsa = {} # stores virtual loss

        self.Es = {}  # stores env.getenvEnded ended for board s

    def getActionProb(self, state, env, temp=0):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state and return action probability.
        Args:
            state : numpy array which represent state  (7,21)
            env : envorionment(deepcube)

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)   , list type
        """
        self.initial_env = env
        self.env = copy.deepcopy(env)

        # numMCTSSims : number of MCTS simulations
        for i in range(self.numMCTSSim):
            self.search(state)
            self.env = copy.deepcopy(self.initial_env)
            self.Lsa = {}

        s = np.array2string(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.action_dim)] #(6,) list  number of times that (s,a) was visited

        if temp == 0:
            #greedy(by Nsa)
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        elif temp == -1:
            #greedy(by Qsa)
            action_Q = [self.Qsa[(s, a)] if (s, a) in self.Qsa else -float('inf') for a in range(self.action_dim)]
            bestA = action_Q.index(max(action_Q))
            probs = [0] * self.action_dim
            probs[bestA] = 1
            return probs



        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state, env_ended = False):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Args:
            state : numpy array which represents state  (7,21)
            env_ended : Boolean type 

        Returns:
            v: value of the current state (if state is terminal, return 1)
        """

        s = np.array2string(state)

        if s not in self.Es:
            self.Es[s] = env_ended    

        if self.Es[s]:
            # terminal node is found
            return 1

        if s not in self.Ps:
            # leaf node is found
            v, self.Ps[s] = self.model.predict(state)
            self.Ns[s] = 0
            return v

        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.action_dim):
            if (s, a) in self.Qsa:
                if (s,a) in self.Lsa:
                    u = (self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])) - self.Lsa[(s,a)]
                else:
                    u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
            else:
                if (s,a) in self.Lsa:
                    u = (self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s])) - self.Lsa[(s,a)]
                else:
                    u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s])

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act

        # update virtual loss
        if (s,a) in self.Lsa:
            self.Lsa[(s,a)] = self.Lsa[(s,a)] + self.virtual_loss_const
        else:
            self.Lsa[(s,a)] = self.virtual_loss_const

        next_s, _, done, _ = self.env.step(a)




        v = self.search(next_s, env_ended=done)

        

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = max(self.Qsa[(s, a)], v)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v





class MCTS2():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, model, cfg):
        self.model = model

        self.children_and_data = dict()

        self.ch_i = 0  # list of next states
        self.p_i = 1  #policy P
        self.s_i = 2  #value W
        self.n_of_v_i = 3  #number of visit N
        self.v_l_i = 4  #virtual loss L
        self.done = 5  # 


        self.loss_constant = cfg['mcts']['virtual_loss_const']
        self.exploration_constant = cfg['mcts']['cpuct']
        self.action_dim = 6

    def train(self, state, env):
        simulation_env = copy.deepcopy(env)


        path_to_leaf, actions_to_leaf, leaf = self.traverse(state, simulation_env)
        reward = self.expand(leaf, simulation_env)
        self.backpropagate(path_to_leaf, actions_to_leaf, reward)


        for i in range(len(self.children_and_data[np.array2string(leaf)][self.ch_i])):
            if self.children_and_data[np.array2string(leaf)][self.done][i]:
                actions_to_leaf.append(i)
                return actions_to_leaf

        return None

    def traverse(self, state, env):
        """
        This function performs one traverse until it finds leaf node.

        Args:
            state : numpy array which represents state  (7,21)

        Returns:
            path_to_leaf : list of states(string) from root node to leaf node(exclude leaf node)
            actions_to_leaf : list of actions(0-5) from root node to leaf node
            current : state of leaf node
        """

        path_to_leaf = []
        actions_to_leaf = []
        current_arr = state
        current = np.array2string(state)
        while True:
            if current not in self.children_and_data or not self.children_and_data[current][self.ch_i]:
                return path_to_leaf, actions_to_leaf, current_arr
            else:
                if sum(self.children_and_data[current][self.n_of_v_i]) == 0: #?
                    action_index = random.randint(0, self.action_dim - 1)
                else:
                    action_index = self.get_most_promising_action_index(current)

                path_to_leaf.append(current)
                actions_to_leaf.append(action_index)
                self.children_and_data[current][self.v_l_i][action_index] += self.loss_constant

                current_arr, _, _, _ = env.step(action_index)
                current = self.children_and_data[current][self.ch_i][action_index]

    def expand(self, state, env):
        """
        This function performs expansion of node from leaf node.

        Args:
            state : state of leaf node, numpy array size (7,21)
            env : deepcube gym environment

        Returns:
            value : state value of leaf node
        """
        value, policy = self.model.predict(state)

        next_states = []
        is_solved = []
        original_env = copy.deepcopy(env)
        for i in range(self.action_dim):
            next_s, _, done, _ = env.step(i)
            next_states.append(np.array2string(next_s))
            is_solved.append(done)
            env = copy.deepcopy(original_env)
            

        self.children_and_data[np.array2string(state)] = (
            next_states,
            policy,
            [-1] * self.action_dim,
            [0] * self.action_dim,
            [0] * self.action_dim,
            is_solved)
        

        return value

    def backpropagate(self, path_to_leaf, actions_to_leaf, reward):
        """
        This function performs backpropagation from leaf to root node.

        Args:
            path_to_leaf : list of states, path of searched nodes
            actions_to_leaf : list of actions, path of searched nodes
            reward : state value of leaf node

        """
        for state_to_leaf, action_to_leaf in zip(list(reversed(path_to_leaf)), list(reversed(actions_to_leaf))):  # 역순으로 가야하는거 아닌가?
            self.children_and_data[state_to_leaf][self.s_i][action_to_leaf] = max(
                self.children_and_data[state_to_leaf][self.s_i][action_to_leaf], reward)

            self.children_and_data[state_to_leaf][self.v_l_i][action_to_leaf] -= 150

            self.children_and_data[state_to_leaf][self.n_of_v_i][action_to_leaf] += 1
            # print(self.children_and_data[state_to_leaf][self.s_i])

    def get_most_promising_action_index(self, state):
        """
        This function give action which is most promising according to paper during searching

        Args:
            path_to_leaf : list of states, path of searched nodes
            actions_to_leaf : list of actions, path of searched nodes
            reward : state value of leaf node
        
        Return:
            index of most promising action

        """

        state_all_actions_number_of_visits = sum(self.children_and_data[state][self.n_of_v_i])
        u_plus_w_a = [0] * self.action_dim
        for i in range(self.action_dim):
            u_st_a = self.exploration_constant \
                     * self.children_and_data[state][self.p_i][i] \
                     * (math.sqrt(state_all_actions_number_of_visits)
                        / (1 + self.children_and_data[state][self.n_of_v_i][i]))
            u_plus_w_a[i] = u_st_a \
                            + self.children_and_data[state][self.s_i][i] \
                            - self.children_and_data[state][self.v_l_i][i]

        return max(range(self.action_dim), key=u_plus_w_a.__getitem__)

    """
    Leaving here as I wrote it for testing to make sure bfs works
    
    def expand_levels(self, state, current_depth, max_depth):
        if current_depth > max_depth:
            return
        direct_children = state.get_direct_children_if_not_solved()
        self.children[state] = direct_children
        for direct_child in direct_children:
            self.expand_levels(direct_child, current_depth + 1, max_depth)
    """

    def bfs(self, state):
        """
        Uncomment and invoke method directly in main with matching number_of_turns and max_depth
        to make sure bfs works
        self.children = dict()
        self.expand_levels(state, 0, 2)
        """

        visited = {state}
        solved = None
        state_to_parent_and_index_from_parent = dict()
        state_to_parent_and_index_from_parent[state] = (None, None)

        queue = deque()
        queue.append(state)
        while len(queue) != 0:
            current = queue.popleft()
            if current.is_solved():
                solved = current
                break

            if current not in self.children_and_data:  # in MCTS not all branches are visited
                continue

            i = 0
            for current_child in self.children_and_data[current][self.ch_i]:
                if current_child not in visited:
                    queue.append(current_child)
                    state_to_parent_and_index_from_parent[current_child] = (current, i)
                    visited.add(current_child)

                i += 1

        if solved is None:
            return None

        current = solved
        reversed_actions_to_leaf = []
        while True:
            pair = state_to_parent_and_index_from_parent[current]
            current = pair[0]
            if current is None:
                break

            reversed_actions_to_leaf.append(pair[1])

        reversed_actions_to_leaf.reverse()
        return reversed_actions_to_leaf