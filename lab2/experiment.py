from genetic_algorithm import GeneticAlgorithm
from problem import evaluate
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def analyze_mutation_prob(
        ga,
        initial_population,
        max_iterations,
        population_size,
        crossover_prob,
        mutation_probs,
        num_repeats
):
    """
    Analyzes impact of mutation_prob on average optimization cost.

    Parameters:
        ga (GeneticAlgorithm): Instance of genetic algorithm with set fitness function.
        initial_population (list): Initial population for the algorithm.
        max_iterations (int): Maximum number of algorithm iterations.
        population_size (int): Population size.
        crossover_prob (float): Constant value of crossover_prob to use in the experiment.
        mutation_probs (list): List of mutation_prob values to test.
        num_repeats (int): Number of repeats for each mutation_prob value.
    """
    print("Starting mutation_prob analysis...")
    average_costs = []

    for mutation_prob in mutation_probs:
        costs = []
        for _ in range(num_repeats):
            _, best_fitness = ga.solve(
                initial_population=initial_population,
                max_iterations=max_iterations,
                population_size=population_size,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob
            )
            costs.append(best_fitness)
        average_cost = np.mean(costs)
        print(
            f"Average cost for mutation_prob={mutation_prob}: {average_cost}")
        average_costs.append(average_cost)

    plt.figure(figsize=(10, 6))
    plt.plot(mutation_probs, average_costs,
             marker='o', linestyle='-', color='b')
    plt.xlabel("Mutation Probability")
    plt.ylabel("Average Best Cost")
    plt.title("Impact of Mutation Probability on Optimization Cost")
    plt.grid(True)
    plt.savefig("mutation_prob_impact.png")
    plt.show()


def analyze_crossover_prob(
        ga,
        initial_population,
        max_iterations,
        population_size,
        crossover_probs,
        mutation_prob,
        num_repeats
):
    """
    Analyzes impact of crossover_prob on average optimization cost.

    Parameters:
        ga (GeneticAlgorithm): Instance of genetic algorithm with set fitness function.
        initial_population (list): Initial population for the algorithm.
        max_iterations (int): Maximum number of algorithm iterations.
        population_size (int): Population size.
        crossover_probs (list): List of crossover_prob values to test.
        mutation_prob (float): Constant value of mutation_prob to use in the experiment.
        num_repeats (int): Number of repeats for each crossover_prob value.
    """
    print("Starting crossover_prob analysis...")
    average_costs = []

    for crossover_prob in crossover_probs:
        costs = []
        for _ in range(num_repeats):
            _, best_fitness = ga.solve(
                initial_population=initial_population,
                max_iterations=max_iterations,
                population_size=population_size,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob
            )
            costs.append(best_fitness)
        average_cost = np.mean(costs)
        print(
            f"Average cost for crossover_prob={crossover_prob}: {average_cost}")
        average_costs.append(average_cost)

    plt.figure(figsize=(10, 6))
    plt.plot(crossover_probs, average_costs,
             marker='o', linestyle='-', color='g')
    plt.xlabel("Crossover Probability")
    plt.ylabel("Average Best Cost")
    plt.title("Impact of Crossover Probability on Optimization Cost")
    plt.grid(True)
    plt.savefig("crossover_prob_impact2.png")
    plt.show()


def analyze_population_size(
        ga,
        initial_population,
        max_iterations,
        population_sizes,
        crossover_prob,
        mutation_prob,
        num_repeats
):
    """
    Analyzes impact of population_size on average optimization cost and standard deviation.

    Parameters:
        ga (GeneticAlgorithm): Instance of genetic algorithm with set fitness function.
        initial_population (list): Initial population for the algorithm.
        max_iterations (int): Maximum number of algorithm iterations.
        population_sizes (list): List of population sizes to test.
        crossover_prob (float): Constant value of crossover_prob to use in the experiment.
        mutation_prob (float): Constant value of mutation_prob to use in the experiment.
        num_repeats (int): Number of repeats for each population_size value.
    """
    print("Starting population_size analysis...")
    average_costs = []
    std_devs = []

    for population_size in population_sizes:
        costs = []
        for _ in range(num_repeats):
            _, best_fitness = ga.solve(
                initial_population=initial_population,
                max_iterations=max_iterations,
                population_size=population_size,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob
            )
            costs.append(best_fitness)
        average_cost = np.mean(costs)
        std_dev = np.std(costs)
        print(
            f"Average cost for population_size={population_size}: {average_cost}, Standard Deviation: {std_dev}")
        average_costs.append(average_cost)
        std_devs.append(std_dev)

    plt.figure(figsize=(10, 6))
    plt.plot(population_sizes, average_costs,
             marker='o', linestyle='-', color='b')
    plt.xlabel("Population Size")
    plt.ylabel("Average Best Cost")
    plt.title("Impact of Population Size on Average Optimization Cost")
    plt.grid(True)
    plt.savefig("population_size_average_cost2.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(population_sizes, std_devs, marker='o', linestyle='-', color='r')
    plt.xlabel("Population Size")
    plt.ylabel("Standard Deviation of Best Cost")
    plt.title("Impact of Population Size on Standard Deviation of Optimization Cost")
    plt.grid(True)
    plt.savefig("population_size_std_dev2.png")
    plt.show()


def calculate_average_best_cost(
        ga,
        initial_population,
        max_iterations,
        population_size,
        crossover_prob,
        mutation_prob,
        num_repeats
):
    """
    Calculates the average best cost over a number of repeats for given genetic algorithm parameters.

    Parameters:
        ga (GeneticAlgorithm): Instance of genetic algorithm with set fitness function.
        initial_population (list): Initial population for the algorithm.
        max_iterations (int): Maximum number of algorithm iterations.
        population_size (int): Population size.
        crossover_prob (float): Probability of crossover.
        mutation_prob (float): Probability of mutation.
        num_repeats (int): Number of repeats for calculating the average.

    Returns:
        float: The average best cost over the specified number of repeats.
    """
    print("Calculating average best cost...")
    costs = []
    for _ in range(num_repeats):
        _, best_fitness = ga.solve(
            initial_population=initial_population,
            max_iterations=max_iterations,
            population_size=population_size,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob
        )
        print("Best fitness", best_fitness)
        costs.append(best_fitness)

    average_cost = np.mean(costs)
    return average_cost


ga = GeneticAlgorithm(fitness_function=evaluate)

initial_population = [np.random.randint(0, 2, 200).tolist() for _ in range(10)]

max_iterations = 1000
population_size = 100
mutation_prob = 0.7
crossover_prob = 0.9
mutation_probs = np.linspace(0, 1, 11)
crossover_probs = np.linspace(0, 1, 11)
population_sizes = [10, 20, 50, 100, 200]
num_repeats = 10

# Analysis of mutation_prob
analyze_mutation_prob(ga, initial_population, max_iterations,
                      population_size, crossover_prob, mutation_probs, num_repeats)

# Analysis of crossover_prob
analyze_crossover_prob(ga, initial_population, max_iterations,
                       population_size, crossover_probs, mutation_prob, num_repeats)

# Analysis of population_size
analyze_population_size(ga, initial_population, max_iterations,
                        population_sizes, crossover_prob, mutation_prob, num_repeats)

# Calculate average best cost for the best hyperparametes
average_cost = calculate_average_best_cost(
    ga=ga,
    initial_population=initial_population,
    max_iterations=1000,
    population_size=100,
    crossover_prob=0.9,
    mutation_prob=0.7,
    num_repeats=20
)
print("Średni najlepszy koszt:", average_cost)

# Run algorithm with given parameteres
best_solution = ga.solve(initial_population, max_iterations=1000,
                         population_size=100, crossover_prob=0.9, mutation_prob=0.7)

print("Best solution: ", best_solution[0])
print("Cost: ", best_solution[1])
