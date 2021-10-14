# import stuff
import numpy
import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import jacks_car_rental

# the model parameters
gamma = 0.9
theta = 1e-4

if __name__ == '__main__':
    j = jacks_car_rental.Jcr(20, 20)
    print(j.state.cars_at_A)
    print(j.state.cars_at_B)
    # all_states = j.get_all_possible_states()
    # print(j.rental_request_probs_A)
    m, reward = j.brownian_movement()
    x = range(0,20)
    y = range(0,20)
    A = np.zeros(shape=(21, 21))
    R = np.zeros(shape=(21, 21))
    for l, prob in m.items():
        # print(loc[0], loc[1], prob)
        A[l[0], l[1]] = prob

    # A = np.where(A>0.01, 1, 0)
    # A = np.array([[0, 0, 0],[1, 1, 2]])
    # sns.heatmap(A)
    # plt.show()
    print(A[9, 9])
    print(A[10, 10])
    print(f"reward: {reward}")
    # for s in all_states:
    # print(s.cars_at_A, s.cars_at_B)

    # Initialization
    V = np.zeros(shape=(21, 21), dtype=float)
    # init the policy. keep in mind that all actions of the policy need to be feasible
    policy = np.zeros(shape=(21, 21), dtype=int)

    # Policy evaluation
    delta = 0.0
    while True:

        # Policy Evaluation
        for i in range(0, 21):
            for j in range(0, 21):
                v = V[i, j]
                s = jacks_car_rental.Jcr(i, j)
                s.move_in_the_night(policy[i, j])
                # according to Bellman equation
                # print(f"cars at A: {s.state.cars_at_A}, cars at B: {s.state.cars_at_B}")
                state_primes, reward = s.brownian_movement()
                for coord, prob in state_primes.items():
                    V[i, j] += prob*gamma*V[coord[0], coord[1]]
                V[i, j] += reward
                # delta = np.max(delta, np.abs(v-V[i,j]))
                delta = max(delta, abs(v - V[i, j])) # np.max(delta, np.abs(v-V[i,j]))

        # Exit condition for policy Evaluation
        print(f'delta: {delta}')
        if delta < theta:
            break



        #sns.heatmap(V)
        #plt.show()

        #    break
        #input("next step?")






