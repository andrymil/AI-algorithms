from solver import Solver
import numpy as np
import random


class GeneticAlgorithm(Solver):
    """
    Class for finding minimum using genetic algorithm.

    Attributes:
        fitness_function (function): the function to optimize
    """

    def __init__(self, fitness_function=None) -> None:
        """
        The constructor for GeneticAlgorithm class.

        Parameters:
            fitness_function (function): the function to optimize
        """
        self.fitness_function = fitness_function

    def get_parameters(self):
        """
        Get the function to optimize.

        Returns:
            function: the function to optimize
        """
        return self.fitness_function

    def solve(self, initial_population, max_iterations, population_size, crossover_prob, mutation_prob):
        """
        Find minimum using genetic algorithm.

        Parameters:
            initial_population (list): the initial population
            max_iterations (int): the maximum number of iterations (generations)
            population_size (int): the number of individuals in the population
            crossover_prob (float): the probability of crossover
            mutation_prob (float): the probability of mutation
        """
        t = 0
        current_population = initial_population
        fitness_scores = self.evaluate_population(current_population)

        best_individual, best_fitness = self.find_best_individual(
            current_population, fitness_scores)

        while t < max_iterations:
            selected_parents = self.selection(
                current_population, fitness_scores)

            children = self.crossover_and_mutate(
                selected_parents, population_size, crossover_prob, mutation_prob)

            children_fitness_scores = self.evaluate_population(children)

            best_child, best_child_fitness = self.find_best_individual(
                children, children_fitness_scores)

            if best_child_fitness < best_fitness:
                best_individual = best_child
                best_fitness = best_child_fitness

            current_population = children
            fitness_scores = children_fitness_scores
            t += 1

        return best_individual, best_fitness

    def evaluate_population(self, population):
        """
        Calculate fitness for each individual in the population.

        Parameters:
            population (list): List of individuals (solutions).

        Returns:
            list: Fitness values for each individual.
        """
        return [self.fitness_function(individual) for individual in population]

    def find_best_individual(self, population, fitness_scores):
        """
        Find the best individual in the population based on fitness scores.

        Parameters:
            population (list): List of individuals.
            fitness_scores (list): Fitness values for each individual.

        Returns:
            tuple: Best individual and its fitness score.
        """
        best_index = np.argmin(fitness_scores)
        return population[best_index], fitness_scores[best_index]

    def selection(self, population, fitness_scores):
        """
        Select parents based on fitness scores.

        Parameters:
            population (list): List of individuals.
            fitness_scores (list): Fitness values for each individual.

        Returns:
            list: Selected parents.
        """
        total_fitness = sum(fitness_scores)
        selection_probs = [score / total_fitness for score in fitness_scores]

        selected_parents = random.choices(
            population, weights=selection_probs, k=len(population) // 2
        )
        return selected_parents

    def crossover_and_mutate(self, parents, population_size, crossover_prob, mutation_prob):
        """
        Perform crossover and mutation on the parents to create a new population.

        Parameters:
            parents (list): List of selected parents.
            population_size (int): The number of individuals in the population.
            crossover_prob (float): The probability of crossover.
            mutation_prob (float): The probability of mutation.

        Returns:
            list: New population after crossover and mutation.
        """
        children = []
        while len(children) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            if random.random() < crossover_prob:
                crossover_point = random.randint(1, len(parent1) - 1)
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                children.extend([child1, child2])
            else:
                children.extend([parent1, parent2])

        children = children[:population_size]

        mutated_children = []
        for child in children:
            mutated_child = [
                gene if random.random() > mutation_prob else 1 - gene
                for gene in child
            ]
            mutated_children.append(mutated_child)

        return mutated_children
