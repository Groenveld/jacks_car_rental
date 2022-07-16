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



    def rent_1D(self, rental_probs, dim):
        self.S_prime = self.init_S()
        reward = 0
        # go trough all the S
        for s, p in self.S.items():
            if p == 0:
                continue
            # else:
                #print(s, p)
            # go trough all the possible transitions in dimension A
            if s[dim] == 0:
                self.S_prime[s] += p
                continue
            else:
                rental_probs[s[dim]] = rental_probs[s[dim]:].sum()
                rental_probs[s[dim]+1:] = 0

            for i, r in enumerate(rental_probs):
                if dim == 0:
                    if s[0]-i < 0:
                        continue
                    self.S_prime[s[0]-i, s[1]] += p*r
                    reward += p*r*i*self.rental_reward_A
                else:
                    if s[1]-i < 0:
                        continue
                    self.S_prime[s[0], s[1]-i] += p*r
                    reward += p*r*i*self.rental_reward_B

            self.S = self.S_prime.copy()

        return reward



    def brownian_1D(self, request_probs, return_probs, cars_at_location, max_cap_at_location):
        """
        :param request_probs:
        :param return_probs:
        :param cars_at_location:
        :param max_cap_at_location:
        :return:
        """
        # print(f"brownian here: cars at location: {cars_at_location}")
        # 1. Rent cars (if possible)
        # if we have no cars, no cars can be requested. hence the probabilities are set all to zero
        if cars_at_location == 0:
            request_probs = np.zeros(shape=request_probs.shape)
        else:
            # the requests exceeding capacity are maped to the max available.
            # e.g. if 3 cars are available, the probabilities for more than 3 are mapped (summed) to three
            # and the probs for 4+ are set to zero
            request_probs[cars_at_location] = request_probs[cars_at_location:].sum()
            request_probs[cars_at_location+1:] = 0
        # 2. return cars (if possible)
        # if the location is full, no cars will be returned
        if  cars_at_location == max_cap_at_location:
            return_probs[0] = 1
            return_probs[1:] = 0
        else:
            if cars_at_location != 0:
                return_probs[max_cap_at_location-cars_at_location] = return_probs[max_cap_at_location-cars_at_location:].sum()
                return_probs[max_cap_at_location-cars_at_location+1:] = 0
        move = dict()
        reward = 0
        for i, p_req in enumerate(request_probs):
            if p_req == 0:
                continue
            reward += p_req*i
            for j, p_ret in enumerate(return_probs):
                if p_ret == 0:
                    continue
                # logging.debug(f'old prob: {move.get(-i+j)}, new prob: {move.get(-i+j, 0) + p_req*p_ret}')
                move[-i+j] = move.get(-i+j, 0) + p_req*p_ret

        for k, v in move.items():
            logging.info(f'{k}: {v}')
        return move, reward

    def get_possible_actions(self):
        possible_actions = []
        # a move of 3 means moving 3 cars from A to B
        for n in range(0, min(self.state.cars_at_A, self.max_move)):
            possible_actions.append(n)
        # a move of -2 means moving 2 cars from B to A
        for n in range(0, min(self.state.cars_at_B, self.max_move)):
            possible_actions.append(-n)
        logging.info(f'Possible actions: {possible_actions}')
        return possible_actions
    def brownian_movement(self):
        """
        there are some fluctuations during the day.
        At the end of the day, some cars are rented and returned at each location.
        A movement consist of movement in A and B
        1. Cars rented at A
        2. Cars returned at A
        3. Cars rented at B
        4. Cars returned at B
        requests and returns are independent
        movements in A and B are independent
        movement delta = (-rent_A + return_A, -rent_B + return-B)
        what is the probability of each movement?
        consider that no more car can be rented than available at location.
        Hence a request of 3 at a location with 2 cars will be mapped to 2
        Hence we need to consider where we are
        A return of 2 at a place with 19 will be mapped to 20
        the movements in A and B are calculated separately in brownian_1D
        :return: p(s'|s)
        """
        move_A, reward_A = self.brownian_1D(
            request_probs=self.rental_request_probs_A,
            return_probs=self.rental_return_probs_A,
            cars_at_location=self.state.cars_at_A,
            max_cap_at_location=self.max_cap_A
        )

        move_B, reward_B = self.brownian_1D(
            request_probs=self.rental_request_probs_B,
            return_probs=self.rental_return_probs_B,
            cars_at_location=self.state.cars_at_B,
            max_cap_at_location=self.max_cap_B
        )
        # return probability of ending at state s' given state s
        # the result is given as dictionary with key (cars_at_a, cars_at_b) and value prob between 0.0 and 1.0
        s_primes = dict()
        weighted_rewards = dict()
        for i_a, p_a in move_A.items():
            for i_b, p_b in move_B.items():
                s_primes[(self.state.cars_at_A+i_a, self.state.cars_at_B+i_b)] = p_a*p_b
        reward = reward_A*self.rental_reward_A+reward_B*self.rental_reward_B

        return s_primes, reward

    def rent_cars_heuristic(self):
        """
        n cars are requested at each location based on the poisson distribution of that location,
        n cars (upon availability) are rented, hence not anymore at the location,
        the reward is 10$ per car rented
        """
        # rent for location A with poisson of A
        request_A = np.random.choice(len(self.rental_request_probs_A), 1, p=self.rental_request_probs_A)
        rent_A = np.min(request_A, self.state.cars_at_A)
        logging.info(f'could rent {rent_A} out of {request_A} cars')
        self.state.cars_at_A = self.state.cars_at_A - rent_A
        # rent for location B with poisson of B
        request_B = np.random.choice(len(self.rental_request_probs_B), 1, p=self.rental_request_probs_B)
        rent_B = np.min(request_B,self.state.cars_at_B)
        logging.info(f'could rent {rent_B} out of {request_B} cars')
        self.state.cars_at_B = self.state.cars_at_B - rent_B
        return rent_A, rent_B

    def return_cars_heuristic(self):
        # Returning cars on A:
        return_A = np.random.choice(len(self.rental_request_probs_A), 1, p=self.rental_return_probs_A)
        parked_cars = self.state.cars_at_A
        self.state.cars_at_A = np.min(self.state.cars_at_A + return_A, self.max_cap_A)

        # Returning cars on B
        return_B = np.random.choice(len(self.rental_request_probs_B), 1, p=self.rental_return_probs_B)
        parked_cars = self.state.cars_at_B
        self.state.cars_at_B = np.min(self.state.cars_at_B + return_B, self.max_cap_B)
        logging.info(f'Already at A: {parked_cars}, '
                     f'parking {return_B} cars '
                     f'resulting in {self.state.cars_at_B} cars. '
                     f'Parking-limit: {self.max_cap_B}')


    def move_in_the_night(self, n):
        """
        moves n cars from A to B, if negative, it means moving from B to A
        it is done trough changing the state of the object
        :return: Nothing
        """
        # check for feasibility of the move:
        assert n < self.max_move, f"tried to move {n} cars where max movement is {self.max_move}"
        assert self.state.cars_at_A - n >= 0, f"cant take away {n} cars from A, bc there are only {self.state.cars_at_A}"
        assert self.state.cars_at_B + n >= 0, f"cant take away {n} cars from B, bc there are only {self.state.cars_at_B}"
        # if all is legal, we execute the moves
        self.state.cars_at_A = self.state.cars_at_A - n
        self.state.cars_at_B = self.state.cars_at_B + n
        # we only have 20 parking slots. If we put more, then cars will be removed from the game (lost)
        if self.state.cars_at_A > self.max_cap_A:
            self.state.cars_at_A = self.max_cap_A
        if self.state.cars_at_B > self.max_cap_B:
            self.state.cars_at_B = self.max_cap_B
