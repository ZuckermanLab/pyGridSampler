import unittest
import numpy as np
from multiprocessing import cpu_count
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from pyGridSampler import grid_sampler as gs
from pyGridSampler import grid_tools as gt


class TestGetMarginalsFromGrid(unittest.TestCase):
    
    def test_get_marginals_from_grid(self):
        # Test with a simple 2D grid
        grid = np.array([[1, 2], [3, 4]])
        expected_output = [[1, 3], [2, 4]]
        self.assertEqual(gt.get_marginals_from_grid(grid), expected_output)

        # Test with a 3D grid
        grid = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        expected_output = [[[1, 5], [3, 7]], [[2, 6], [4, 8]]]
        self.assertEqual(gt.get_marginals_from_grid(grid), expected_output)

        # Test with an empty grid
        grid = np.array([])
        expected_output = []
        self.assertEqual(gt.get_marginals_from_grid(grid), expected_output)

    
def func(x, args):
    return -np.sum(x**2)  # Negative of the squared Euclidean distance from the origin

    
class TestEvalGridPoints(unittest.TestCase):
    def test_eval_grid_points(self):
        # Set up inputs
        grid, _ = gt.gen_grid_points(n_dim=2, x_bounds=[(-1, 1), (-1, 1)], grid_resolution=10)
        n_processes = cpu_count()

        # Evaluate in parallel
        log_likelihoods_par, rel_prob_par, weights_par, ess_par = gt.eval_grid_points(grid, func, args=(np.array([0, 0]),), n_processes=n_processes)

        # Evaluate serially
        log_likelihoods_ser = [func(x, args=(np.array([0, 0]),)) for x in grid]
        rel_prob_ser = gt.calc_rel_p(gt.calc_rel_logp(log_likelihoods_ser))
        weights_ser = gt.calc_norm_weights(rel_prob_ser)
        ess_ser = gt.calc_ess(rel_prob_ser)

        # Verify that the results are the same
        np.testing.assert_allclose(log_likelihoods_ser, log_likelihoods_par, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(rel_prob_ser, rel_prob_par, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(weights_ser, weights_par, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ess_ser, ess_par, rtol=1e-6, atol=1e-6)

        # Assert that the parallel output has the expected shape
        self.assertEqual(np.array(log_likelihoods_par).shape, (grid.shape[0],))
        self.assertEqual(np.array(rel_prob_par).shape, (grid.shape[0],))
        self.assertEqual(np.array(weights_par).shape, (grid.shape[0],))
        self.assertIsInstance(ess_par, float)

        # check values of outputs
        self.assertAlmostEqual(np.sum(weights_par), 1.0)
        self.assertGreaterEqual(ess_par, 1.0)
        

class TestGenGridPoints(unittest.TestCase):
    
    def test_gen_grid_points(self):
        # generate grid
        grid, x_spacing = gt.gen_grid_points(n_dim=2, x_bounds=[(-1, 1), (-1, 1)], grid_resolution=11)
        
        # check types and shapes of outputs
        self.assertIsInstance(grid, np.ndarray)
        self.assertIsInstance(x_spacing, list)
        self.assertEqual(grid.shape, (121, 2))
        self.assertEqual(len(x_spacing), 2)
        
        # check values of outputs
        np.testing.assert_almost_equal(x_spacing, np.array([0.2, 0.2]))
        np.testing.assert_almost_equal(grid[0], np.array([-1., -1.]))
        np.testing.assert_almost_equal(grid[-1], np.array([1., 1.]))


class TestUpdateXSpacing(unittest.TestCase):
    
    def test_update_x_spacing(self):
        # Set up inputs
        x_spacing = [0.2, 0.2, 0.2]
        div_amount = 2

        # Call function being tested
        new_x_spacing = gt.update_x_spacing(x_spacing, div_amount)

        # Define expected output
        expected_x_spacing = [0.1, 0.1, 0.1]

        # Verify that the results are the same
        np.testing.assert_allclose(new_x_spacing, expected_x_spacing, rtol=1e-6, atol=1e-6)

        # Assert that the output has the expected shape and type
        self.assertEqual(len(new_x_spacing), len(x_spacing))
        self.assertIsInstance(new_x_spacing[0], float)


class TestReduceGridPoints(unittest.TestCase):
    
    def test_reduce_grid_points(self):
        # Define inputs
        grid = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0], [0.1, 0.1]])
        weights = np.array([0.2, 0.3, 0.1, 0.4])
        delta = 0.2
        
        # Calculate expected reduced grid
        total_weight = np.sum(weights)
        sorted_indices = np.argsort(weights)[::-1]  # Get the indices that sort the weights from highest to lowest
        cumulative_weights = np.cumsum(weights[sorted_indices]) / total_weight  # Calculate the cumulative weights
        threshold_index = np.argmax(cumulative_weights >= 1-delta)  # Find the index of the first weight whose cumulative sum is >= 1-delta
        indices_to_keep = sorted_indices[:threshold_index+1]  # Get the indices of the grid points to keep
        expected_reduced_grid = grid[indices_to_keep]  # Calculate the expected reduced grid
        
        # Call the function to get the actual reduced grid
        reduced_grid = gt.reduce_grid_points(grid, weights, delta)
        
        # Assert that the expected and actual reduced grids are equal
        np.testing.assert_allclose(reduced_grid, expected_reduced_grid, rtol=1e-6, atol=1e-6)


class TestCheckGridBoundary(unittest.TestCase):
    
    def test_check_grid_boundary_remove_points(self):
        # Define inputs
        grid = np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.5], [1.1, 1.2], [-0.1, -0.2]])
        x_bounds = [(0, 1), (0, 1)]
        expected_updated_grid = np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.5]])
        
        # Call function
        updated_grid = gt.check_grid_boundary(grid, x_bounds)
        
        # Check shapes match
        self.assertEqual(updated_grid.shape, expected_updated_grid.shape)
        
        # Check array values match within tolerance
        np.testing.assert_allclose(updated_grid, expected_updated_grid, rtol=1e-6, atol=1e-6)
        
        # Check all points in updated_grid are within the specified bounds
        for point in updated_grid:
            for dim, bound in enumerate(x_bounds):
                self.assertGreaterEqual(point[dim], bound[0])
                self.assertLessEqual(point[dim], bound[1])

    def test_check_grid_boundary_no_remove(self):
        grid = np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.5]])
        x_bounds = [(0, 1), (0, 1)]
        updated_grid = gt.check_grid_boundary(grid, x_bounds)
        expected_updated_grid = np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.5]])
        np.testing.assert_allclose(updated_grid, expected_updated_grid, rtol=1e-6, atol=1e-6)
        
        # Check that the shape of the updated grid is the same as the original grid
        self.assertEqual(updated_grid.shape, grid.shape)
        
        # Check that the updated grid is a different array object than the original grid
        self.assertIsNot(updated_grid, grid)

class TestAddGridPoints(unittest.TestCase):
    
    def test_add_grid_points(self):
        grid = np.array([[0.1, 0.2], [0.2, 0.3]])
        x_bounds = [(0.0, 1.0), (0.0, 1.0)]
        x_shifts = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        x_spacing = [0.1, 0.1]
        expanded_grid = gt.add_grid_points(grid, x_bounds, x_shifts, x_spacing)
        expected_expanded_grid = np.array([[0.1, 0.2], [0.2, 0.3], [0.2, 0.2], [0.3, 0.3], [0.0, 0.2], [0.1, 0.3], [0.1, 0.1], [0.2, 0.4]])

        print("Expanded Grid: \n", expanded_grid)
        print("Expected Expanded Grid: \n", expected_expanded_grid)

        self.assertEqual(expanded_grid.shape[0], expected_expanded_grid.shape[0], "Number of rows do not match")
        self.assertEqual(expanded_grid.shape[1], expected_expanded_grid.shape[1], "Number of columns do not match")
        
        # Check if all elements in expanded grid are within bounds
        for i in range(expanded_grid.shape[0]):
            for j in range(expanded_grid.shape[1]):
                self.assertTrue(expanded_grid[i,j] >= x_bounds[j][0], "Element out of lower bound")
                self.assertTrue(expanded_grid[i,j] <= x_bounds[j][1], "Element out of upper bound")

        np.testing.assert_allclose(expanded_grid, expected_expanded_grid, rtol=1e-6, atol=1e-6, err_msg="Arrays are not equal")




if __name__ == '__main__':
    unittest.main()

