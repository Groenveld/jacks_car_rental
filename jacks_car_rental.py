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
    assert np.sum(res) == 1, "sum pf probs != 1"
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
    def __init__(self, cars_at_A, cars_at_B):
        self.cars_at_A = cars_at_A
        self.cars_at_B = cars_at_B


class Jcr:
    def __init__(self, cars_at_A=0, cars_at_B=0):
        # consts
        self.cost_rental = -10 # negative cost means profit
        self.cost_move = 2
        self.rental_request_rate_A = 3  # orig: 3
        self.rental_request_rate_B = 4  # orig: 4
        self.return_rate_A = 1  # orig: 3
        self.return_rate_B = 1  # orig: 2
        self.max_cap_A = 20
        self.max_cap_B = 20
        self.max_move = 5
        # the state is defined as the amount of cars at A and B
        self.state = JcrState(cars_at_A, cars_at_B)
        # calculate rental and return prob distribution once
        self.rental_request_probs_A = poisson(self.rental_request_rate_A, self.max_cap_A+1)
        self.rental_request_probs_B = poisson(self.rental_request_rate_B, self.max_cap_B+1)
        self.rental_return_probs_A = poisson(self.return_rate_A, self.max_cap_A+1)
        self.rental_return_probs_B = poisson(self.return_rate_B, self.max_cap_B+1)
        # states of the game defined as tuple (cars_at_A, cars_at_B)

    def get_all_possible_states(self):
        states = []
        for a in range(0, self.max_cap_A+1):
            for b in range(0, self.max_cap_B + 1):
                states.append(JcrState(a, b))
        logging.info(f'all possible states: {states}')
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


    def brownian_1D(self, request_probs, return_probs, cars_at_location, max_cap_at_location):
        """
        :param request_probs:
        :param return_probs:
        :param cars_at_location:
        :param max_cap_at_location:
        :return:
        """

        request_probs[cars_at_location] = request_probs[cars_at_location:].sum()
        request_probs[cars_at_location+1:] = 0
        if max_cap_at_location == cars_at_location:
            return_probs[0] = 1
            return_probs[1:] = 0
        else:
            return_probs[max_cap_at_location-cars_at_location] = return_probs[max_cap_at_location-cars_at_location:].sum()
            return_probs[max_cap_at_location-cars_at_location+1:] = 0
        move = dict()
        reward = dict()
        for i, p_req in enumerate(request_probs):
            if p_req == 0:
                continue
            for j, p_ret in enumerate(return_probs):
                if p_ret == 0:
                    continue
                # logging.debug(f'old prob: {move.get(-i+j)}, new prob: {move.get(-i+j, 0) + p_req*p_ret}')
                move[-i+j] = move.get(-i+j, 0) + p_req*p_ret
                reward[-i+j] = reward.get(-i+j, 0) + (p_req*p_ret)*i*self.cost_rental
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

        for i_a, r_a in reward_A.items():
            for i_b, r_b in reward_B.items():
                weighted_rewards[(self.state.cars_at_A+i_a, self.state.cars_at_B+i_b)] = r_a+r_b

        return s_primes, weighted_rewards

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
