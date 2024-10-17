from solver import Solver
from autograd import grad as gradient
import autograd.numpy as np


# Gradient Descent Solver
class GradientDescent(Solver):
    def __init__(self, function=None):
        self.function = function

    def get_parameters(self):
        return self.function

    # Find a path to minimum using gradient descent
    def solve(self, x0, step_size, num_iterations):
        grad = gradient(self.function)
        x = x0
        history = [x0]
        for i in range(num_iterations):
            grad_val = grad(x)
            x = x - step_size * grad_val
            history.append(x)
        return np.array(history)
