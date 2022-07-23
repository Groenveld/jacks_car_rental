import jacks_car_rental as jcr
import matplotlib.pyplot as plt
import plotly.graph_objects as go

if __name__ == '__main__':
    plt.close('all')
    jack = jcr.Jcr()

    for i in range(5):
        print(i)
        jack.policy_evaluation()
        if jack.policy_improvement():
            break

    jcr.to_draw_something(jack.P)
    jcr.to_draw_something(jack.V)
    # fig = go.Figure(data=go.Contour(
    #     z=jack.P)
    # )
    # fig.show()


