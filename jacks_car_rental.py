import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import helper_math as hm

logging.basicConfig(filename='jcr_app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def get_center_of_mass(A):
    """
    returns the x,y position where the distribution would be in balance when sticking on a needle
    """
    mass_x = 0
    mass_y = 0
    dim_x, dim_y = A.shape
    for i in range(dim_x):
        for j in range(dim_y):
            mass_x += A[i, j] * i
            mass_y += A[i, j] * j
    return np.array([mass_x, mass_y])


def to_draw_something(A, title="draw", annot=True):
    ax = sns.heatmap(A, square=True, linewidths=.01, annot=annot)
    ax.invert_yaxis()
    # cbar_kws={"shrink": .5})
    # mask=mask, cmap=cmap, vmax=.3, center=0,
    plt.title(f"{title}")
    plt.show()


def get_s_prime(a, b):
    return a.reshape(-1, 1) @ b.reshape(1, -1)


class Jcr:
    def __init__(self):
        logging.info(f"initializing class")
        # consts
        self.gamma = 0.9
        self.theta = 1e-6
        self.rental_reward = {'A': 10, 'B': 10}
        self.request_rate = {'A': 3, 'B': 4}  # orig: 3/4
        self.return_rate = {'A': 3, 'B': 2}  # orig: 3/2
        self.max_cap = {'A': 20, 'B': 20}
        self.max_move = 5
        self.cost_move = 2
        self.S = np.zeros((self.max_cap['A'], self.max_cap['B']))
        # calculate rental and return prob distribution once
        margin_treatment = 'rescale'
        self.rental_dict = {'A': hm.get_wall_vec_dict(self.request_rate['A'], self.max_cap['A'] + 1, margin_treatment=margin_treatment),
                            'B': hm.get_wall_vec_dict(self.request_rate['B'], self.max_cap['B'] + 1, margin_treatment=margin_treatment)}
        self.return_dict = {'A': hm.get_wall_vec_dict(self.return_rate['A'], self.max_cap['A'] + 1, margin_treatment='sum'),
                            'B': hm.get_wall_vec_dict(self.return_rate['B'], self.max_cap['B'] + 1, margin_treatment='sum')}
        self.index_dict = hm.get_index_vec_dict(max(self.max_cap['A'], self.max_cap['B']) + 1)
        # S of the game defined as matrix of probabilities [cars_at_A, cars_at_B], probability)
        self.V = np.zeros((self.max_cap['A'] + 1, self.max_cap['B'] + 1))
        self.P = np.zeros((self.max_cap['A'] + 1, self.max_cap['B'] + 1), dtype=numpy.int)
        self.check_init_policies()

    def check_init_policies(self):
        for i in range(self.P.shape[0]):
            for j in range(self.P.shape[1]):
                assert self.is_feasible_action(i, j, self.P[i, j]), f"action not feasible: {i, j, self.P[i, j]}"

    def apply_action(self, i, j, a):
        cost = self.cost_move * np.abs(a)
        return i - a, j + a, cost

    def apply_policy(self, i, j):
        a = self.P[i, j]
        cost = self.cost_move * a
        return i - a, j + a, cost

    def rent_cars1D(self, dim, i):
        walled = self.rental_dict[dim][i]
        reward = self.rental_reward[dim] * np.sum(self.index_dict[i] * walled)
        return walled, reward

    def rent_cars(self, i, j):
        rented_a, reward_a = self.rent_cars1D('A', i)
        rented_b, reward_b = self.rent_cars1D('B', j)
        return np.flip(rented_a), np.flip(rented_b), reward_a + reward_b

    def return_cars1D(self, dim, vec):
        res = np.zeros(self.max_cap[dim] + 1)
        for i, v in enumerate(vec):
            res[i:] += v * self.return_dict[dim][self.max_cap[dim] - i]
        return res

    def return_cars(self, a, b):
        returned_a = self.return_cars1D('A', a)
        returned_b = self.return_cars1D('B', b)
        return returned_a, returned_b

    def policy_evaluation(self):
        print(f"called policy evaluation")
        # value_s = self.get_center_of_mass()
        assert self.V.shape == (self.max_cap['A'] + 1, self.max_cap['B'] + 1)
        # to_draw_something(self.V)
        sweep = 0
        while True:
            delta = 0
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    v = self.V[i, j]
                    i_moved, j_moved, cost = self.apply_policy(i, j)
                    rented_a, rented_b, reward = self.rent_cars(i_moved, j_moved)
                    returned_a, returned_b = self.return_cars(rented_a, rented_b)
                    self.V[i, j] = (reward - cost
                                    + self.gamma *
                                    np.sum(np.multiply(get_s_prime(returned_a, returned_b), self.V)))
                    delta = max(delta, np.abs(v - self.V[i, j]))

            logging.info(f"sweep: {sweep}, delta: {delta}")
            # to_draw_something(self.V)
            if delta < self.theta:
                print(f"sweep: {sweep}; delta: {delta}")
                # to_draw_something(self.V)
                return
            sweep += 1

    def policy_improvement(self):
        print(f"called policy improvement")
        policy_stable = True
        for i in range(self.P.shape[0]):
            for j in range(self.P.shape[1]):
                old_action = self.P[i, j]
                possible_actions = self.get_feasible_actions(i, j)
                action_values = np.zeros(len(possible_actions))
                for a_index, a in enumerate(possible_actions):
                    i_moved, j_moved, cost = self.apply_action(i, j, a)
                    rented_a, rented_b, reward = self.rent_cars(i_moved, j_moved)
                    returned_a, returned_b = self.return_cars(rented_a, rented_b)
                    action_values[a_index] = (reward - cost
                                              + self.gamma * np.sum(np.multiply(get_s_prime(returned_a, returned_b), self.V)))
                best_action = possible_actions[np.argmax(action_values)]
                self.P[i, j] = best_action
                if old_action != best_action:
                    policy_stable = False
        # to_draw_something(self.P)
        if policy_stable:
            print(f" policy is stable!!!")
            print(np.amin(self.V), np.amax(self.V))
            return True
        else:
            return False

    def get_feasible_actions(self, i, j):
        a = self.max_move
        # we compare the
        vec1 = np.array([a, a, a])
        vec2 = np.array([i, self.max_cap['B'] - j, self.max_move])
        a_max = np.min(np.minimum(vec1, vec2))
        a = -self.max_move
        vec1 = np.array([-a, -a, -a])
        vec2 = np.array([j, self.max_cap['A'] - i, self.max_move])
        a_min = -np.min(np.minimum(vec1, vec2))
        return np.arange(a_min, a_max + 1)

    def is_feasible_action(self, i, j, a):
        if np.abs(a) > self.max_move:
            return False
        if i - a < 0 or i - a > self.max_cap['A']:
            return False
        if j + a < 0 or j + a > self.max_cap['B']:
            return False
        return True

    def get_feasible_actions_old(self, i, j):
        # min_action = self.is_feasible_action2(i, j, -20)
        # max_action = self.is_feasible_action2(i, j, 20)
        # return min_action, max_action+1
        feasible_actions = []
        for a in np.arange(-self.max_move, self.max_move + 1):
            if self.is_feasible_action(i, j, a):
                feasible_actions.append(a)
        return feasible_actions
