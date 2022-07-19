import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import math

logging.basicConfig(filename='jcr_app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def poisson(lambd, n, eps=1e-7):
    """
    Returns a vector of poisson distributed probabilities
    :param lambd:
    :param n:       Length of array to be returned
    :param eps:     minimum value of a probability, if less, it gathers the sum of all the following probs as well
    :return:        Array of poisson probs where p(k) = p[k]
    """
    res = np.zeros(n)
    assert lambd>0, "requirement for poisson: lambda>0"

    for i, n in enumerate(range(0, n)):
        p = np.power(lambd, n)/math.factorial(n)*np.exp(-lambd)
        # we want nonnegative probabilities. p might get negative for too big n bc of limited float capabilities
        assert p > 0, "probability must be non-negative -> get bigger epsilon to avoid this error"
        if p > eps:
            res[i] = p
        else:
            res[i] = 1-np.sum(res)
            break
    # assert np.isclose(np.sum(res), 1, atol=10*eps), f"sum of probs != 1 {np.sum(res)}"
    if not np.isclose(np.sum(res), 1, atol=10*eps):
        print(f"sum of probs is just: {np.sum(res)}")
    return res


def plot_poissons(p1, p2):
    absolute_tolerance = 1e-6
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # poisson 1
    ax1.plot(p1, label=f'lambda=3')
    ax1.plot(p2, label=f'lambda=4')
    ax1.set_title('poisson')
    ax1.legend()
    # poisson 2
    ax2.plot(np.log10(1-np.cumsum(p1)))
    ax2.plot(np.log10(1-np.cumsum(p2)))
    ax2.hlines(y=math.log10(absolute_tolerance), xmin=0, xmax=20, colors='r')
    ax2.set_title('log(1 - cumsum of poisson)')
    ax2.set_xticks(range(0, len(p1)))
    ax2.set_xticklabels([str(n) for n in range(0, len(p1))])
    # show plot
    plt.show()


def wall_vector(vec, n):
    """
    when a vector of length e.g. 10 is passed and n is 5, the elements 6:10 will be added to element 5
    the returning vec has length 5
    :param vec: incoming vector
    :param n: length of the returning vec
    :return: vec with len n
    """
    # assert n > 0, "walled vec must have len > 0"
    # assert n <= len(vec), f"walled vec of len {n} must be smaller than original vector of {len(vec)}"
    walled_vec = np.zeros(n)
    walled_vec[0:n-1] = vec[0:n-1]
    walled_vec[-1] = np.sum(vec[n-1:])
    # assert np.isclose(sum(vec), np.sum(walled_vec), 1e-7), f"vec and walled vec sum is not the same: {np.sum(vec)} {np.sum(walled_vec)} {vec} {walled_vec}"
    return walled_vec

def make_prob_matrix(lambda_x, lambda_y, shape):
    # create a walled column vector
    p_x = wall_vector(poisson(lambd=lambda_x, n=20), shape[0]).reshape(-1, shape[0])
    # create a walled row vector
    p_y = wall_vector(poisson(lambd=lambda_y, n=20), shape[1]).reshape(shape[1], -1)
    # create the walled matrix
    return p_y@p_x

def make_prob_matrix_dict(lambda_x, lambda_y, max_shape):
    d = dict()
    for x in range(1, max_shape[0]+1):
        for y in range(1, max_shape[1]+1):
            d[(x,y)] = make_prob_matrix(lambda_x, lambda_y, (x, y))
    return d
