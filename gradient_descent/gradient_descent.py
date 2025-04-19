from solver import Solver
from autograd import grad as gradient
import autograd.numpy as np


class GradientDescent(Solver):
    """
    Class for finding a path to minimum using gradient descent.

    Attributes:
        function (function): the function to optimize
    """
    def __init__(self, function=None):
        """
        The constructor for GradientDescent class.

        Parameters:
            function (function): the function to optimize
        """
        self.function = function

    def get_parameters(self):
        """
        Get the function to optimize.

        Returns:
            function: the function to optimize
        """
        return self.function

    def solve(self, x0, step_size, num_iterations):
        """
        Find a path to minimum using gradient descent.

        Parameters:
            x0 (float): the initial value
            step_size (float): the step size
            num_iterations (int): the number of iterations
        """
        grad = gradient(self.function)
        x = x0
        history = [x0]
        for i in range(num_iterations):
            grad_val = grad(x)
            x = x - step_size * grad_val
            history.append(x)
        return np.array(history)
