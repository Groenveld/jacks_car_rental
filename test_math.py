import unittest
import helper_math
import numpy as np
absolute_tolerance = 1e-6
lambd1 = 3
lambd2 = 4
n = 20

class TestHelperMath(unittest.TestCase):

    def test_plot_poisson(self):
        p1 = helper_math.poisson(lambd1, n)
        p2 = helper_math.poisson(lambd2, n)
        helper_math.plot_poissons(p1, p2)
        self.assertTrue(True)

    def test_poisson(self):
        vec_sum1 = np.sum(helper_math.poisson(lambd1, n))
        vec_sum2 = np.sum(helper_math.poisson(lambd2, n))
        self.assertTrue(np.isclose(vec_sum1, 1, atol=absolute_tolerance))
        self.assertTrue(np.isclose(vec_sum2, 1, atol=absolute_tolerance))

    def test_wall_vector(self):
        vec = np.array([1, 1, 1, 1, 1])
        n = 3
        walled_vec = helper_math.wall_vector(vec, n)
        expected_walled_vec = np.array([1, 1, 3])
        self.assertTrue(np.equal(walled_vec, expected_walled_vec).all())

    def test_make_prob_matrix(self):
        test_shape = (20, 20)
        A = helper_math.make_prob_matrix(lambda_x=3, lambda_y=4, shape=test_shape)
        self.assertEqual(A.shape, test_shape)
        self.assertTrue(np.isclose(A.sum().sum(), 1, atol=absolute_tolerance))



if __name__ == '__main__':
    unittest.main()