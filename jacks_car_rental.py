import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import helper_math as hm

logging.basicConfig(filename='jcr_app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class Jcr:
    def __init__(self, init_S=None):
        # consts
        self.gamma = 0.9
        self.rental_reward_A = 10 # negative cost means profit
        self.rental_reward_B = 10 # negative cost means profit
        self.cost_move = -2
        self.rental_request_rate_A = 3  # orig: 3
        self.rental_request_rate_B = 4  # orig: 4
        self.return_rate_A = 1  # orig: 3
        self.return_rate_B = 1  # orig: 2
        self.max_cap_A = 20
        self.max_cap_B = 20
        self.max_move = 5
        # calculate rental and return prob distribution once
        self.rental_request_probs_A = hm.poisson(self.rental_request_rate_A, self.max_cap_A+1)
        self.rental_request_probs_B = hm.poisson(self.rental_request_rate_B, self.max_cap_B+1)
        self.rental_return_probs_A = hm.poisson(self.return_rate_A, self.max_cap_A+1)
        self.rental_return_probs_B = hm.poisson(self.return_rate_B, self.max_cap_B+1)
        # S of the game defined as matrix of probabilities [cars_at_A, cars_at_B], probability)
        if init_S:
            self.S = init_S
        else:
            self.S = self.init_zero_S()
        self.S_prime = self.init_zero_S()
        self.V = np.zeros((self.max_cap_A+1, self.max_cap_B+1))
        self.P = np.zeros((self.max_cap_A+1, self.max_cap_B+1), dtype=numpy.int)-10

    def init_zero_S(self):
        return np.zeros((self.max_cap_A+1, self.max_cap_B+1))

    def to_draw(self):
        sns.heatmap(self.S)
        plt.show()

    def to_draw_something(self, A):
        sns.heatmap(A)
        plt.show()

    def get_center_of_mass(self):
        """
        returns the x,y position where the distribution would be in balance when sticking on a needle
        """
        mass_x = 0
        mass_y = 0
        dim_x, dim_y = self.S.shape
        for i in range(dim_x):
            for j in range(dim_y):
                mass_x += self.S[i, j] * i
                mass_y += self.S[i, j] * j
        return np.array([mass_x, mass_y])

    def rent_cars(self, i, j):
        walled_a = hm.wall_vector(self.rental_request_probs_A, i + 1)
        print(walled_a)
        walled_b = hm.wall_vector(self.rental_request_probs_B, j + 1)
        reward = (self.rental_reward_A*np.sum(np.arange(len(walled_a))*walled_a) +
                self.rental_reward_B*np.sum(np.arange(len(walled_b))*walled_b))
        # get probabilities of going somewhere
        p_a = np.flip(walled_a).reshape(-1, 1)
        p_b = np.flip(walled_b).reshape(1, -1)
        # print("coord: ", i, j, "probs: ", p_i, p_j, "matrix affected: \n", A_to_be_altered)
        S_after_rent = np.zeros((self.max_cap_A + 1, self.max_cap_B + 1))
        S_after_rent[0:(i + 1), 0:(j + 1)] += (p_a @ p_b).T
        return S_after_rent, reward

    def return_cars(self, A):
        S_after_return = np.zeros((self.max_cap_A + 1, self.max_cap_B + 1))
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                walled_a = hm.wall_vector(self.rental_return_probs_A, A.shape[0] - i)
                walled_b = hm.wall_vector(self.rental_return_probs_B, A.shape[1] - j)
                S_after_return[i:,j:] += A[i,j]*(walled_a.reshape(-1,1)@walled_b.reshape(1,-1))

        return S_after_return

    def apply_policy(self, A):
        S_after_policy = np.zeros((self.max_cap_A+1, self.max_cap_B+1))
        print(f"shape of S_after_policy:  {S_after_policy.shape}")

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                p = self.P[i,j]
                # p might be positive or negative
                # check that i+p doesnt exceed 20 or gets below 0
                print(f"policy for coord {i,j} is {p}")
                print(f"the shape of input matrix: {A.shape}")
                print(f"i+p= {i+p}")
                print(f"j-p= {j-p}")
                print(f"{min(i+p, A.shape[0]-1)}")
                print(f"{max(j-p, 0)}")

                # print(i + p, A.shape[0], np.min(i+p, A.shape[0]))
                # print(i - p, 0, np.min(i-p, 0))
                if p >= 0:
                    S_after_policy[min(i+p, A.shape[0]-1), max(j-p, 0)] += A[i, j]
                else:
                    S_after_policy[max(i+p, 0), min(j-p, A.shape[1]-1)] += A[i, j]
        return S_after_policy



    def policy_evaluation(self):
        """
        go trough S and apply the rental poisson. as we can not rent out more than we have, the wall_vector
        function will limit the requests exceeding the stock to the stock available.
        the parking can at max be empty, not negative
        :return:
        """
        print(f"called policy evaluation")
        # value_s = self.get_center_of_mass()
        assert self.S.shape == (self.max_cap_A+1, self.max_cap_B+1)
        delta = 0
        while True:
            for i in range(self.S.shape[0]):
                for j in range(self.S.shape[1]):
                    v = self.V[i,j]
                    S_after_rent, reward = self.rent_cars(i, j)
                    S_after_return = self.return_cars(S_after_rent)
                    self.V[i,j] = reward + self.gamma*np.multiply(S_after_return, self.V)
                    delta = np.max(delta, np.abs(v-self.V[i, j]))
            if delta < 1e-6:
                return

        value_s_prime = self.get_center_of_mass()
        print(f"reward: {value_s_prime-value_s}")
        # print(self.get_center_of_mass())
        # print(f"sum after renting: {A_prime.sum()}")

