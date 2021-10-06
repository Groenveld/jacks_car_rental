# import stuff
import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import jacks_car_rental

# the model parameters
gamma = 0.9

if __name__ == '__main__':
    j = jacks_car_rental.Jcr(10, 10)
    print(j.state.cars_at_A)
    print(j.state.cars_at_B)
    # all_states = j.get_all_possible_states()
    # print(j.rental_request_probs_A)
    m, r = j.brownian_movement()
    x = range(0,20)
    y = range(0,20)
    A = np.zeros(shape=(21, 21))
    R = np.zeros(shape=(21, 21))
    for l, prob in m.items():
        # print(loc[0], loc[1], prob)
        A[l[0], l[1]] = prob

    for l, reward in r.items():
        R[l[0], l[1]] = reward
    # A = np.where(A>0.01, 1, 0)
    # A = np.array([[0, 0, 0],[1, 1, 2]])
    sns.heatmap(R)
    plt.show()
    print(A[9, 9])
    print(A[10, 10])
    print(R[9, 9])
    print(R[7, 6])
    print(R[6, 7])
    print(R.sum())
    # for s in all_states:
    # print(s.cars_at_A, s.cars_at_B)

