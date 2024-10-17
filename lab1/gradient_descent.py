import os
from solver import Solver
from autograd import grad as gradient
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


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


# Function f(x)
def f(x):
    return 0.5 * x**4 + x


# Function g(x1, x2)
def g(X):
    x1, x2 = X
    term1 = np.exp(- (x1**2 + (x2 + 1)**2))
    term2 = np.exp(- ((x1 - 1.75)**2 + (x2 + 2)**2))
    return 1 - 0.6 * term1 - 0.4 * term2


# Plotting gradient descent for f(x)
def plot_gradient_descent_f(x_init, step_size, num_iterations, show_plot=True, save_plot=True):
    # Real minimum of f(x)
    real_minimum = -1 / 2**(1/3)

    # Perform gradient descent
    gradient_descent = GradientDescent(f)
    history_f = gradient_descent.solve(x_init, step_size, num_iterations)

    # Create the plot
    x_vals = np.linspace(-10, 10, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='f(x) = 0.5x^4 + x')
    plt.scatter(history_f, f(history_f), color='red',
                label='Gradient Descent steps', zorder=5)
    plt.scatter(history_f[-1], f(history_f[-1]), color='blue',
                s=100, label="Estimated Minimum", zorder=5)
    plt.axvline(x=real_minimum, color='green', linestyle='--',
                label="Real Minimum")
    plt.plot(history_f, f(history_f), color='red', linestyle='--', alpha=0.6)

    plt.title('Gradient Descent on f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    # Save the plot
    if save_plot:
        save_path = f'function_f/iterations_{num_iterations}/alpha_{step_size}/plot_x0_{x_init}.png'
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path)
        print(f"Plot saved as: {save_path}")

    # Display the plot
    if show_plot:
        plt.show()


# Plotting gradient descent for g(x) (2D)
def plot_gradient_descent_g(x_init, step_size, num_iterations, show_plot=True, save_plot=True):
    # Perform gradient descent
    gradient_descent = GradientDescent(g)
    history_g = gradient_descent.solve(x_init, step_size, num_iterations)

    # Create the contour plot
    x1_vals = np.linspace(-2, 4, 400)
    x2_vals = np.linspace(-4, 4, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = np.array([[g([x1, x2]) for x1, x2 in zip(row1, row2)]
                 for row1, row2 in zip(X1, X2)])

    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z, levels=50, cmap='viridis')
    history_g = np.array(history_g)
    plt.scatter(history_g[:, 0], history_g[:, 1],
                color='red', label='Gradient Descent steps', zorder=5)
    plt.plot(history_g[-1, 0], history_g[-1, 1], 'o',
             color='blue', markersize=10, label='Last Point (Min)')
    plt.plot(history_g[:, 0], history_g[:, 1],
             color='red', linestyle='--', alpha=0.6)

    plt.title('Gradient Descent on g(x1, x2)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

    # Save the plot
    if save_plot:
        save_path = f'function_g/iterations_{num_iterations}/alpha_{step_size}/plot_init_{x_init[0]}_{x_init[1]}_2d.png'
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path)
        print(f"Plot saved as: {save_path}")

    # Display the plot
    if show_plot:
        plt.show()


# Plotting gradient descent for g(x) (3D)
def plot_gradient_descent_g_3d(x_init, step_size, num_iterations, show_plot=True, save_plot=True):
    # Run gradient descent
    gradient_descent = GradientDescent(g)
    history = gradient_descent.solve(x_init, step_size, num_iterations)

    # Create the 3D plot
    x1_vals = np.linspace(-2, 2, 100)
    x2_vals = np.linspace(-3.5, 0.5, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = g([X1, X2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    # Plot the gradient descent steps
    x1_hist = history[:, 0]
    x2_hist = history[:, 1]
    z_hist = g([x1_hist, x2_hist])
    ax.scatter(x1_hist, x2_hist, z_hist, color='r',
               label='Gradient Descent steps', s=50)
    ax.scatter(x1_hist[-1], x2_hist[-1], z_hist[-1],
               color='b', s=100, label='Estimated minimum')

    ax.set_title('Gradient Descent on g(x1, x2)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('g(x1, x2)')
    ax.legend()

    # Save the plot
    if save_plot:
        save_path = f'function_g/iterations_{num_iterations}/alpha_{step_size}/plot_init_{x_init[0]}_{x_init[1]}_3d.png'
        folder = os.path.dirname(save_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(save_path)
        print(f"Plot saved as: {save_path}")

    # Display the plot
    if show_plot:
        plt.show()


# Run the experiments for function f(x)
def experiment_f(step_sizes, x_initial_values, num_iterations):
    for x_init in x_initial_values:
        for step_size in step_sizes:
            for num_iter in num_iterations:
                print(
                    f"\nRunning gradient descent for f(x) with x_init={x_init},"
                    + f"step_size={step_size}, num_iterations={num_iter}")
                plot_gradient_descent_f(
                    x_init, step_size, num_iter, show_plot=False, save_plot=True)


# Run the experiments for function g(x1, x2)
def experiment_g(step_sizes, x_initial_values, num_iterations):
    for x_init in x_initial_values:
        for step_size in step_sizes:
            for num_iter in num_iterations:
                print(
                    f"\nRunning gradient descent for g(x1, x2) with x_init={x_init},"
                    + f"step_size={step_size}, num_iterations={num_iter}")
                plot_gradient_descent_g(
                    x_init, step_size, num_iter, show_plot=False, save_plot=True)
                plot_gradient_descent_g_3d(
                    x_init, step_size, num_iter, show_plot=False, save_plot=True)


def experiment_gradient_descent():
    step_sizes_f = [0.005, 0.01, 0.1]
    step_sizes_g = [0.1, 1, 5]

    x_initial_values_f = [-4.0, 2.0, 7.0]

    x_initial_values_g = [
        np.array([-1.0, -3.0]), np.array([0.0, 0.0]), np.array([1.5, -3.0])]

    num_iterations = [20, 200]

    print("Experimenting with function f(x)...")
    experiment_f(step_sizes_f, x_initial_values_f, num_iterations)

    print("\nExperimenting with function g(x1, x2)...")
    experiment_g(step_sizes_g, x_initial_values_g, num_iterations)


# Run the experiments
experiment_gradient_descent()

# Estimate minimum of f(x)
gradient_descent = GradientDescent(f)
history_f = gradient_descent.solve(2.0, 0.01, 500)
x_minimum_f = history_f[-1]
print("Real minimum of f(x): -0.5952753945 in -0.7937005259")
print(f"Estimated minimum of f(x): {f(x_minimum_f)} in {x_minimum_f}")

# Estimate minimum of g(x)
gradient_descent = GradientDescent(g)
history_g = gradient_descent.solve(np.array([-1.0, -3.0]), 1, 200)
x_minimum_g = history_g[-1]
print(f"Estimated minimum of g(x): {g(x_minimum_g)} in {x_minimum_g}")


# # Draw gradient descent for f(x)
# x_init_f = 2.0
# step_size_f = 0.01
# num_iterations_f = 200
# show_plot_f = True
# save_plot_f = False
# plot_gradient_descent_f(x_init_f, step_size_f,
#                         num_iterations_f, show_plot_f, save_plot_f)

# # Draw gradient descent for g(x)
# x_init_g = np.array([-1.0, -3.0])
# step_size_g = 1
# num_iterations_g = 200
# show_plot_g = True
# save_plot_g = True
# plot_gradient_descent_g(x_init_g, step_size_g,
#                         num_iterations_g, show_plot_g, save_plot_g)
# plot_gradient_descent_g_3d(x_init_g, step_size_g,
#                            num_iterations_g, show_plot_g, save_plot_g)
