import jacks_car_rental as jcr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

if __name__ == '__main__':
    plt.close('all')
    jack = jcr.Jcr()
    policy_history = []
    for i in range(9):
        print(i)
        jack.policy_evaluation()
        policy_stable = jack.policy_improvement()
        policy_history.append(jack.P)
        jcr.to_draw_something(jack.P, annot=True)
        if policy_stable:
            break
        if i >= 2:
            if (policy_history[i] == policy_history[i-2]).all():
                break

    jcr.to_draw_something(jack.P, annot=True)
    jcr.to_draw_something(jack.V, annot=False)
    print(f"Value functions extrema: {np.amin(jack.V), np.amax(jack.V)}")


