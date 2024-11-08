import pygmo as pg
import numpy as np
import argparse
import pandas as pd
# Step 1: Define a custom problem class for x^2
class MyProblem:
    def __init__(self):
        self.dim = 1  # The problem is 1-dimensional
    
    # Fitness function: the objective we are minimizing (x^2)
    def fitness(self, x):
        return [ objective_function(x[0], x[1], x[2]) ]  # The function to minimize is x^2
    
    # Get bounds for the decision variable (range for x)
    def get_bounds(self):
        return ([-100, -100, -100], [100, 100, 100])  # x can range between -10 and 10

def objective_function(x, y, z):
    return (x**2 + y**2 + z**2) + np.sin(5 * x) * np.sin(5 * y) * np.sin(5 * z)
    
# Initialize the argument parser

problem = pg.problem(MyProblem())

algo = pg.algorithm(pg.pso(gen=50))

# Step 4: Create a population (size 10)
pop = pg.population(problem, size=5)

# Step 5: Evolve the population to solve the problem
pop = algo.evolve(pop)

# Step 6: Output the results
print("Best solution found: x =", pop.champion_x)
print("Minimum value of x^2: f(x) =", pop.champion_f)