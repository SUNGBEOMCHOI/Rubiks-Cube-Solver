import logging
import math
import random
import copy
from collections import deque

import numpy as np

from utils import get_env_config

EPS = 1e-8

log = logging.getLogger(__name__)

class MCTS():
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
        self.value_min = cfg['mcts']['value_min']
        self.cube_size = cfg['test']['cube_size']
        _, self.action_dim = get_env_config(self.cube_size)

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
            [self.value_min] * self.action_dim,
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
        return 