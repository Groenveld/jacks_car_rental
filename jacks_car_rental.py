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


def to_draw_something(A):
    sns.heatmap(A)
    plt.show()


class Jcr:
    def __init__(self, init_S=None):
        print(f"initializing class")
        # consts
        self.gamma = 0.9
        self.rental_reward_A = 10
        self.rental_reward_B = 10
        self.cost_move = -2
        self.rental_request_rate_A = 2  # orig: 3
        self.rental_request_rate_B = 2  # orig: 4
        self.return_rate_A = 0.00001  # orig: 3
        self.return_rate_B = 1  # orig: 2
        self.max_cap_A = 20
        self.max_cap_B = 20
        self.max_move = 5
        # calculate rental and return prob distribution once
        self.rental_request_probs_A = hm.poisson(self.rental_request_rate_A, self.max_cap_A + 1)
        self.rental_request_probs_B = hm.poisson(self.rental_request_rate_B, self.max_cap_B + 1)
        self.rental_return_probs_A = hm.poisson(self.return_rate_A, self.max_cap_A + 1)
        self.rental_return_probs_B = hm.poisson(self.return_rate_B, self.max_cap_B + 1)
        # S of the game defined as matrix of probabilities [cars_at_A, cars_at_B], probability)
        if init_S:
            self.S = init_S
        else:
            self.S = np.zeros((self.max_cap_A + 1, self.max_cap_B + 1))
            print(self.S.shape)
        self.S_prime = self.init_zero_S()
        self.V = np.zeros((self.max_cap_A + 1, self.max_cap_B + 1))
        self.P = np.zeros((self.max_cap_A + 1, self.max_cap_B + 1), dtype=numpy.int)
        self.check_init_policies()

    def check_init_policies(self):
        for i in range(self.S.shape[0]):
            for j in range(self.S.shape[1]):
                assert self.is_feasible_action(i, j, self.P[i, j]), f"action not feasible: {i, j, self.P[i, j]}"

    def init_zero_S(self):
        return np.zeros((self.max_cap_A + 1, self.max_cap_B + 1))

    def to_draw(self):
        sns.heatmap(self.S)
        plt.show()

    def rent_cars(self, i, j):
        walled_a = hm.wall_vector(self.rental_request_probs_A, i + 1)
        walled_b = hm.wall_vector(self.rental_request_probs_B, j + 1)
        reward = (self.rental_reward_A * np.sum(np.arange(len(walled_a)) * walled_a) +
                  self.rental_reward_B * np.sum(np.arange(len(walled_b)) * walled_b))
        # get probabilities of going somewhere
        p_a = np.flip(walled_a).reshape(-1, 1)
        p_b = np.flip(walled_b).reshape(1, -1)
        # print("coord: ", i, j, "probs: ", p_i, p_j, "matrix affected: \n", A_to_be_altered)
        S_after_rent = np.zeros((self.max_cap_A + 1, self.max_cap_B + 1))
        S_after_rent[0:(i + 1), 0:(j + 1)] += (p_a @ p_b)
        return S_after_rent, reward

    def return_cars(self, A):
        S_after_return = np.zeros((self.max_cap_A + 1, self.max_cap_B + 1))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                walled_a = hm.wall_vector(self.rental_return_probs_A, A.shape[0] - i)
                walled_b = hm.wall_vector(self.rental_return_probs_B, A.shape[1] - j)
                S_after_return[i:, j:] += A[i, j] * (walled_a.reshape(-1, 1) @ walled_b.reshape(1, -1))
        return S_after_return

    def is_feasible_action2(self, i, j, a):
        # we compare the
        if a >= 0:
            vec1 = np.array([a, a, a])
            vec2 = np.array([i, self.max_cap_B - j, self.max_move])
            return np.min(np.minimum(vec1, vec2))
        else:
            vec1 = np.array([-a, -a, -a])
            vec2 = np.array([j, self.max_cap_A - i, self.max_move])
            return np.min(np.minimum(vec1, vec2))

    def is_feasible_action(self, i, j, a):
        if np.abs(a) > self.max_move:
            return False
        if i - a < 0 or i - a > self.max_cap_A:
            return False
        if j + a < 0 or j + a > self.max_cap_B:
            return False
        return True

    def get_feasible_actions(self, i, j):
        # TODO dont append lists but calc min and max and create range(min, max)
        feasible_actions = []
        for a in np.array(-self.max_move, self.max_move + 1):
            if self.is_feasible_action(i, j, a):
                feasible_actions.append(a)
        return feasible_actions

    def apply_policy(self, i, j):
        a = self.P[i, j]
        cost = self.cost_move * a
        return i - a, j + a, cost

    def policy_evaluation(self):
        """
        go trough S and apply the rental poisson. as we can not rent out more than we have, the wall_vector
        function will limit the requests exceeding the stock to the stock available.
        the parking can at max be empty, not negative
        :return:
        """
        print(f"called policy evaluation")
        # value_s = self.get_center_of_mass()
        assert self.S.shape == (self.max_cap_A + 1, self.max_cap_B + 1)
        delta = 0
        to_draw_something(self.V)
        while True:
            print(f"delta: {delta}")
            for i in range(self.S.shape[0]):
                for j in range(self.S.shape[1]):
                    v = self.V[i, j]
                    # print(i, j, v)
                    i_moved, j_moved, cost = self.apply_policy(i, j)
                    # print(i_moved, j_moved, cost)
                    S_after_rent, reward = self.rent_cars(i_moved, j_moved)
                    # to_draw_something(S_after_rent)
                    S_after_return = self.return_cars(S_after_rent)
                    # to_draw_something(S_after_return)
                    self.V[i, j] = reward - cost + self.gamma * np.sum(np.multiply(S_after_return, self.V))
                    # print(self.V[i, j])
                    # print(i,j, reward, self.V[i, j])
                    delta = max(delta, np.abs(v - self.V[i, j]))
                to_draw_something(self.V)
                print(delta)
            if delta < 1e-6:
                return

        value_s_prime = get_center_of_mass()
        print(f"reward: {value_s_prime - value_s}")
        # print(self.get_center_of_mass())
        # print(f"sum after renting: {A_prime.sum()}")
