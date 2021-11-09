import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import math

logging.basicConfig(filename='jcr_app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def poisson(lambd, max_n, eps=1e-7):
    res = np.zeros(max_n)
    if lambd == 0:
        return res
    for i, n in enumerate(range(0, max_n)):
        p = np.power(lambd, n)/math.factorial(n)*np.exp(-lambd)
        # we want nonnegative probabilities. p might get negative for too big n bc of limited float capabilities
        assert p > 0, "probability must be non-negative -> get bigger epsilon to avoid this error"
        if p < eps:
            res[i] = 1-np.sum(res)
            break
        else:
            res[i] = p
    assert np.isclose(np.sum(res), 1, atol=10*eps), f"sum pf probs != 1 {np.sum(res)}"
    return res


def plot_poissons(p1, p2):
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # poisson 1
    ax1.plot(p1, label=f'lambda=3')
    ax1.plot(p2, label=f'lambda=4')
    ax1.set_title('poisson')
    ax1.legend()
    # poisson 2
    ax2.plot(np.cumsum(p1))
    ax2.plot(np.cumsum(p2))
    ax2.set_title('cumsum of poisson')
    ax2.set_xticks(range(0, len(p1)))
    ax2.set_xticklabels([str(n) for n in range(0, len(p1))])
    # show plot
    plt.show()


class JcrState:
    def __init__(self, cars_at_A, cars_at_B, prob):
        self.cars_at_A = cars_at_A
        self.cars_at_B = cars_at_B
        self.prob = prob

class JcrStates:
    def __init__(self, max_cap_a, max_cap_b, init_states=None):
        if init_states:
            self.states = init_states
        else:
            self.states = self.init_zero(max_cap_a, max_cap_b)

    def init_zero(self, max_cap_a, max_cap_b):
        self.states = []
        for a in range(0, max_cap_a+1):
            for b in range(0, max_cap_b + 1):
                self.states.append(JcrState(a, b, 0.0))


def to_print(d):
    for s, p in d.items():
        print(s, p)


def to_draw(d, max_a, max_b):
    A = np.zeros((max_a + 1, max_b + 1))
    for s, p in d.items():
        A[s[0], s[1]] = p
    sns.heatmap(A)
    plt.show()

def get_center_of_mass(d):
    mass_x = 0
    mass_y = 0
    for s, p in d.items():
        mass_x += s[0]*p
        mass_y += s[1]*p
    return mass_x, mass_y



class Jcr:
    def __init__(self, init_states=None):
        # consts
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
        self.rental_request_probs_A = poisson(self.rental_request_rate_A, self.max_cap_A+1)
        self.rental_request_probs_B = poisson(self.rental_request_rate_B, self.max_cap_B+1)
        self.rental_return_probs_A = poisson(self.return_rate_A, self.max_cap_A+1)
        self.rental_return_probs_B = poisson(self.return_rate_B, self.max_cap_B+1)
        # states of the game defined as dict[cars_at_A, cars_at_B], probability)
        # self.states = JcrStates(init_states, self.max_cap_A, self.max_cap_B)
        self.states = self.init_states()
        self.states_prime = self.init_states()

    def init_states(self, init_states_dict=None):
        if init_states_dict:
            return init_states_dict
        else:
            states = dict()
            for i in range(0, self.max_cap_A+1):
                for j in range(0, self.max_cap_B+1):
                    states[(i, j)] = 0.0
        return states


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

    def rent_cars(self):
        r_a = self.rent_1D(self.rental_request_probs_A, dim=0)
        r_b = self.rent_1D(self.rental_request_probs_B, dim=1)
        return r_a+r_b

    def return_cars(self):
        self.return_1D(self.rental_return_probs_A, dim=0)
        self.return_1D(self.rental_return_probs_B, dim=1)

    def return_1D(self, return_probs, dim):
        self.states_prime = self.init_states()
        for s, p in self.states.items():
            if p == 0:
                continue
            if dim == 0:
                if s[dim] == self.max_cap_A:
                    self.states_prime[s] += p
                else:
                    return_probs[self.max_cap_A-s[dim]] = return_probs[self.max_cap_A-s[dim]:].sum()
                    return_probs[self.max_cap_A-s[dim] + 1:] = 0
            else:
                if s[dim] == self.max_cap_B:
                    self.states_prime[s] += p
                else:
                    return_probs[self.max_cap_B-s[dim]] = return_probs[self.max_cap_B-s[dim]:].sum()
                    return_probs[self.max_cap_B-s[dim] + 1:] = 0

            for i, r in enumerate(return_probs):
                if dim == 0:
                    if s[0]+i > self.max_cap_A:
                        #print(f"this should be 0: {r}")
                        continue
                    self.states_prime[s[0]+i, s[1]] += p*r
                else:
                    if s[1]+i > self.max_cap_B:
                        continue
                    self.states_prime[s[0], s[1]+i] += p*r

            self.states = self.states_prime.copy()



    def rent_1D(self, rental_probs, dim):
        self.states_prime = self.init_states()
        reward = 0
        # go trough all the states
        for s, p in self.states.items():
            if p == 0:
                continue
            # else:
                #print(s, p)
            # go trough all the possible transitions in dimension A
            if s[dim] == 0:
                self.states_prime[s] += p
                continue
            else:
                rental_probs[s[dim]] = rental_probs[s[dim]:].sum()
                rental_probs[s[dim]+1:] = 0

            for i, r in enumerate(rental_probs):
                if dim == 0:
                    if s[0]-i < 0:
                        continue
                    self.states_prime[s[0]-i, s[1]] += p*r
                    reward += p*r*i*self.rental_reward_A
                else:
                    if s[1]-i < 0:
                        continue
                    self.states_prime[s[0], s[1]-i] += p*r
                    reward += p*r*i*self.rental_reward_B

            self.states = self.states_prime.copy()

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
