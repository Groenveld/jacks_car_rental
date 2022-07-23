import jacks_car_rental as jcr
import numpy as np

def test_eval():
    jack = jcr.Jcr()
    s, t = 10, 10
    jack.S[s, t] = 1.0
    jack.P[s, t] = 0
    jcr.to_draw_something(jack.S, 'init')
    i_moved, j_moved, cost = jack.apply_policy(s, t)
    jack2 = jcr.Jcr()
    jack2.S[i_moved, j_moved] = 1
    jcr.to_draw_something(jack2.S, 'after moving')
    print(f"i, j, cost after applying policy: {i_moved, j_moved, cost}")
    rented_a, rented_b, reward = jack.rent_cars(i_moved, j_moved)
    jcr.to_draw_something(jcr.get_s_prime(rented_a, rented_b), 'after renting')
    returned_a, returned_b = jack.return_cars(rented_a, rented_b)
    jcr.to_draw_something(jcr.get_s_prime(returned_a, returned_b), 'after returning')


def test_feasible_actions():
    jack = jcr.Jcr()
    for s in range(21):
        for t in range(21):
            a = jack.get_feasible_actions(s, t)
            b = jack.get_feasible_actions_old(s, t)
            if (a!=b).any():
                print(s, t, a, b)

test_eval()