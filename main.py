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
    j = jacks_car_rental.Jcr()
    print("INIT")
    j.states[(2, 2)] = 1

    print(sum(j.states.values()))
    jacks_car_rental.to_draw(j.states, 20, 20)
    for i in range(10):
        r = j.rent_cars()
        print(j.states[0, 0])
        print(j.states[0, 1])
        print(j.states[1, 0])
        print(j.states[1, 1])
        # print(f"round {i}: reward: {r}")
        #j.return_cars()
        jacks_car_rental.to_draw(j.states, 20, 20)

    # print(j.states[(10,10)]) #
    # print(j.rental_request_probs_A[0])
    # print(j.rental_request_probs_B[0])



    # Initialization
    V = np.zeros(shape=(21, 21), dtype=float)
    V_prime = V.copy()
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
                if (i == 21) & (j == 21):
                    pass
                    print(sum(state_primes.values()))
                new_contribution_sum = 0
                for coord, prob in state_primes.items():
                    new_contribution = prob*gamma*V[coord[0], coord[1]]
                    if (i == 20) & (j == 20):
                       new_contribution_sum += new_contribution
                    V_prime[i, j] += new_contribution
                if (i == 20) & (j == 20):
                    print(f"new_contribution_sum: {new_contribution_sum}")
                V_prime[i, j] += reward
                # delta = np.max(delta, np.abs(v-V[i,j]))
                delta = max(delta, abs(v - V_prime[i, j])) # np.max(delta, np.abs(v-V[i,j]))
        V = V_prime
        print(V[20, 20])
        # Exit condition for policy Evaluation
        print(f'delta: {delta}')
        if delta < theta:
            break
        sns.heatmap(V)
        plt.show()
        #    break
        input("next step?")






