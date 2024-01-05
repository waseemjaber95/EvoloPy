import random
import numpy
import math
import numpy as np
from solution import solution
import time

def GGO(objf, lb, ub, dim, PopSize, Max_iter, population):
    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    print('GGO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    goose = list(population.copy()[0])

    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###    ###

    # Initialize GGO parameters
    r1 = random.random()  # r1 is a random number in [0,1]
    r2 = random.random()  # r2 is a random number in [0,1]
    r3 = random.random()  # r3 is a random number in [0,1]
    r4 = random.random()  # r4 is a random number in [0,1]
    r5 = random.random()  # r5 is a random number in [0,1]
    w = random.uniform(0, 2)  # w is a random number in [0,2]
    w1 = random.uniform(0, 2)  # w1 is a random number in [0,2]
    w2 = random.uniform(0, 2)  # w2 is a random number in [0,2]
    w3 = random.uniform(0, 2)  # w3 is a random number in [0,2]
    w4 = random.uniform(0, 2)  # w4 is a random number in [0,2]
    l = random.uniform(-1, 1)  # l is a random number in [-1,1]
    D = 2  # Sample value for D

    # calculate the objective function for each individual
    fitness = np.zeros(len(goose))
    for i, individual in enumerate(goose):
        fitness[i] = objf(np.array([individual]))
    # Get the index of the minimum fitness value
    min_fitness_index = np.argmin(fitness)
    # Access the corresponding individual in the population
    best_individual = goose[min_fitness_index]

    #initializa previous_best_individual and previous_previous_best_individual
    previous_best_individual, previous_previous_best_individual = None, None

    # Update solutions in exploration group(n1) and exploitation group(n2)
    n = len(goose)
    split_index = math.floor(n / 2)
    n1 = goose[:split_index]
    n2 = goose[split_index:]

    # Main loop
    for t in range(0, Max_iter):
        # Initialize GGO parameters
        a = 2 * (1 - t / Max_iter)  # a decreases linearly from 2 to 0
        A = 2 * a * r1 - a
        C = 2 * r2
        z = 1 - (t / Max_iter) ** 2  # Equation (3)
        b = 2  # b is a constant
        Xflock1 = best_individual


        ### Operations on Exploration group(n1) ###
        # loop through each individual in n1 (exploration group)
        for i in range(0, len(n1)):
            if t % 2 == 0:
                if r3 < 0.5:
                    if abs(A) < 1:
                        n1[i] = best_individual - A * np.abs(C * best_individual - n1[i])  # Equation (1)
                    else:
                        # Select three random search agents Xpaddle1, Xpaddle2, and Xpaddle3
                        if len(n1) >= 3:
                            selected_individuals = random.sample(n1, 3)
                            Xpaddle1, Xpaddle2, Xpaddle3 = selected_individuals
                        else:
                            Xpaddle1, Xpaddle2, Xpaddle3 = random.randint(0,1000), random.randint(0,1000), random.randint(0,1000)

                        # Update z by the exponential form
                        z = 1 - (t/Max_iter) ** 2 # Equation (3)

                        # Update position of current search agent
                        n1[i] = w1 * Xpaddle1 + z * w2 * (Xpaddle2 - Xpaddle3) + (1-z) * w3 * (n1[i] - Xpaddle1) # Equation (2)
                else:
                    # Update position of current search agent
                    n1[i] = w4 * np.abs(best_individual - n1[i]) * np.exp(b * l) * np.cos(2 * np.pi * l) + 2 * w1 * (r4 + r5) * best_individual  # Equation (4)
            else:
                # Update individual positions
                n1[i] = n1[i] + D * (1 + z) * w * (n1[i] - Xflock1)


        ### Operations on Exploitation group(n2) ###
        # Select three random sentries: Xsentry1, Xsentry2, and Xsentry3
        if len(n2) >= 3:
            selected_individuals = random.sample(n2, 3)
            Xsentry1, Xsentry2, Xsentry3 = selected_individuals
        else:
            Xsentry1, Xsentry2, Xsentry3 = random.randint(0,1000), random.randint(0,1000), random.randint(0,1000)

        # loop through each individual in n2 (exploitation group)
        for i in range(0, len(n2)):
            if t % 2 == 0:
                # calculate X1, X2, and X3
                X1 = Xsentry1 - A * np.abs(C * Xsentry1 - n2[i])  # Equation (5)
                X2 = Xsentry2 - A * np.abs(C * Xsentry2 - n2[i])  # Equation (5)
                X3 = Xsentry3 - A * np.abs(C * Xsentry3 - n2[i])  # Equation (5)

                # Update individual positions
                n2[i] = (X1 + X2 + X3) / 3  # Equation (6)
            else:
                # Update position of current search agent
                n2[i] = n2[i] + D * (1 + z) * w * (n2[i] - Xflock1)


        ### Update parameters ###
        r1 = random.random()  # r1 is a random number in [0,1]
        r2 = random.random()  # r2 is a random number in [0,1]
        r3 = random.random()  # r3 is a random number in [0,1]
        r4 = random.random()  # r4 is a random number in [0,1]
        r5 = random.random()  # r5 is a random number in [0,1]
        w = random.uniform(0, 2)  # w is a random number in [0,2]
        w1 = random.uniform(0, 2)  # w1 is a random number in [0,2]
        w2 = random.uniform(0, 2)  # w2 is a random number in [0,2]
        w3 = random.uniform(0, 2)  # w3 is a random number in [0,2]
        w4 = random.uniform(0, 2)  # w4 is a random number in [0,2]
        l = random.uniform(-1, 1)  # l is a random number in [-1,1]


        ### calculate the objective function for each individual ###
        goose = np.concatenate((n1, n2))
        for i, individual in enumerate(goose):
            fitness[i] = objf(np.array([individual]))
        # Get the index of the minimum fitness value
        min_fitness_index = np.argmin(fitness)
        # Store in the previous_previous_best_individual and previous_best_individual
        previous_previous_best_individual = previous_best_individual
        previous_best_individual = best_individual
        # Access the corresponding individual in the population
        best_individual = goose[min_fitness_index]


        ### Adjust beyond the search space solutions ###
        if (best_individual == previous_best_individual) and (best_individual == previous_previous_best_individual):
            split_index = split_index + 1
            n1 = goose[:split_index]
            n2 = goose[split_index:]

        Convergence_curve[t] = best_individual
        print(["At iteration " + str(t) + " the best fitness is " + str(best_individual)])

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GGO"
    s.bestIndividual = best_individual
    s.objfname = objf.__name__

    return s
