import random
import time

import numpy as np

rng = np.random.default_rng()
import numpy as np
import math


# This function is to initialize the Vulture population.
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]
    return X


# Sort fitness.
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


# Sort the position of the Vulture according to fitness.
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


# Boundary detection function.
def BorderCheck1(X, lb, ub, dim):
    for j in range(dim):
        if X[j] < lb[j]:
            X[j] = ub[j]
        elif X[j] > ub[j]:
            X[j] = lb[j]
    return X


def rouletteWheelSelection(x):
    CS = np.cumsum(x)
    Random_value = random.random()
    index = np.where(Random_value <= CS)
    index = sum(index)
    return index


def random_select(Pbest_Vulture_1, Pbest_Vulture_2, alpha, betha):
    probabilities = [alpha, betha]
    index = rouletteWheelSelection(probabilities)
    if index.all() > 0:
        random_vulture_X = Pbest_Vulture_1
    else:
        random_vulture_X = Pbest_Vulture_2

    return random_vulture_X


def exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound):
    if random.random() < p1:
        current_vulture_X = random_vulture_X - (abs((2 * random.random()) * random_vulture_X - current_vulture_X)) * F
    else:
        current_vulture_X = (random_vulture_X - (F) + random.random() * (
                    (upper_bound - lower_bound) * random.random() + lower_bound))
    return current_vulture_X


def exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, variables_no,
                 upper_bound, lower_bound):
    if abs(F) < 0.5:

        if random.random() < p2:

            A = Best_vulture1_X - ((np.multiply(Best_vulture1_X, current_vulture_X)) / (
                        Best_vulture1_X - current_vulture_X ** 2)) * F
            B = Best_vulture2_X - (
                        (Best_vulture2_X * current_vulture_X) / (Best_vulture2_X - current_vulture_X ** 2)) * F
            current_vulture_X = (A + B) / 2
        else:
            current_vulture_X = random_vulture_X - abs(random_vulture_X - current_vulture_X) * F * levyFlight(
                variables_no)

    if random.random() >= 0.5:
        if random.random() < p3:
            current_vulture_X = (abs((2 * random.random()) * random_vulture_X - current_vulture_X)) * (
                        F + random.random()) - (random_vulture_X - current_vulture_X)

        else:
            s1 = random_vulture_X * (random.random() * current_vulture_X / (2 * math.pi)) * np.cos(current_vulture_X)
            s2 = random_vulture_X * (random.random() * current_vulture_X / (2 * math.pi)) * np.sin(current_vulture_X)
            current_vulture_X = random_vulture_X - (s1 + s2)
    return current_vulture_X


# eq (18)
def levyFlight(d):
    beta = 3 / 2
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(1, d) * sigma
    v = np.random.randn(1, d)
    step = u / abs(v) ** (1 / beta)
    o = step
    return o


def AVOA(X, fun, lb, ub, Max_iter):
    time_start = time.time()
    alpha = 0.8
    betha = 0.2
    p1 = 0.6
    p2 = 0.4
    p3 = 0.6
    Gama = 2.5
    pop, dim = X.shape[0], X.shape[1]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    fitness, sortIndex = SortFitness(fitness)  # Sort the fitness values of African Vultures
    X = SortPosition(X, sortIndex)  # Sort the African Vultures population based on fitness
    GbestScore = fitness[0]  # Stores the optimal value for the current iteration.
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = X[0, :]
    Curve = np.zeros([Max_iter, 1])
    Xnew = np.zeros([pop, dim])
    # Main iteration starts here
    for t in range(Max_iter):
        Pbest_Vulture_1 = X[0, :]  # location of Vulture (First best location Best Vulture Category 1)
        Pbest_Vulture_2 = X[1, :]  # location of Vulture (Second best location Best Vulture Category 1)
        t3 = np.random.uniform(-2, 2, 1) * (
                    (np.sin((math.pi / 2) * (t / Max_iter)) ** Gama) + np.cos((math.pi / 2) * (t / Max_iter)) - 1)
        z = random.randint(-1, 0)
        # F= (2*random.random()+1)*z*(1-(t/Max_iter))+t3
        P1 = (2 * random.random() + 1) * (1 - (t / Max_iter)) + t3
        F = P1 * (2 * random.random() - 1)
        # For each vulture Pi
        for i in range(pop):
            current_vulture_X = X[i, :]
            random_vulture_X = random_select(Pbest_Vulture_1, Pbest_Vulture_2, alpha,
                                             betha)  # select random vulture using eq(1)
            if abs(F) >= 1:
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, ub, lb)  # eq (16) & (17)

            else:
                current_vulture_X = exploitation(current_vulture_X, Pbest_Vulture_1, Pbest_Vulture_2, random_vulture_X,
                                                 F, p2, p3, dim, ub, lb)  # eq (10) & (13)

            Xnew[i, :] = current_vulture_X[0]
            Xnew[i, :] = BorderCheck1(Xnew[i, :], lb, ub, dim)
            tempFitness = fun(Xnew[i, :])
            # Update local best solution
            if tempFitness <= fitness[i]:
                fitness[i] = tempFitness
                X[i, :] = Xnew[i, :]
        Ybest, index = SortFitness(fitness)
        X = SortPosition(X, index)
        # Update global best solution
        if Ybest[0] <= GbestScore:
            GbestScore = Ybest[0]
            GbestPositon[0, :] = X[index[0], :]
        # print(GbestPositon)
        Curve[t] = GbestScore
    Time = time.time() - time_start
    return GbestScore, Curve, GbestPositon, Time
