'''
====================================================================================================
Title:  OHM Term Project Code - Simulated Annealing Functions
Author: Divayum Gupta, 17IM30010, Department of Industrial and Systems Engineering, IIT Kharagpur
Mail:   divayumgupta@gmail.com
Date:   23 May 2020
====================================================================================================
'''
'''
VARIABLE NAMES

PARAMETERS OF BANK LENDING (refer Fig 1 of research paper)
Loan Age (alpha) - age
Loan Size - L
Loan Type(phi) - ltype
Credit Rating - rating
Credit Limit - lim
Loan Interest Rate(rL) - interest
Expected Loan Loss(lambda) - loss
Deposit Rate - rD
Reserve Ratio - K
Financial institutions's Deposit - D
Customer Transaction Rate - rT
Number of 'good' customers - N
Pre-determined Institutional Cost(delta) - IC
****************************************************************************************************
VARIABLES IN EVALUATING FITNESS FUNCTION (refer section 3.3)
Loan Revenue(nu) - lrev
Loan Cost(mu) - lcost
Total Transaction Cost(omega) - tcost
Institutional Transaction Cost - T
Cost of Demand Deposit(beta) - costDD
****************************************************************************************************
OPTIMIZATION PARAMETERS FOR SIMULATED ANNEALING
Reduction factor - c
Total Number of Temperature Reductions - total_reductions
Number of iterations at one temperature - n_iters
****************************************************************************************************
RESULT VARIABLES FOR SIMULATED ANNEALING
Individual solution - new_solution or old_solution
Solution fitness value - new_fit or old_fit
All solutions ever generated and their fitness values - results
'''

#IMPORTING LIBRARIES
import numpy as np 
import pandas as pd 
import math
import random

#FUNCTION TO EVALUATE FITNESS OF EACH SOLUTION
def sa_fitness_eval(solution, N, df, IC, K, D, rD, rT):
    lrev = 0    #nu
    lcost = 0   #mu
    tcost = 0   #omega
    sumloss = 0 #sumLambda
    costDD = 0  #beta
    sumloan = 0

    for j in range(N):
        if solution[j] == 1:
            sumloan += df['Loan Size'][j]

    for j in range(N):
        if solution[j] == 1:
            lrev += df['Interest Rate'][j]*df['Loan Size'][j] - df['Loan Loss'][j]
            lcost += df['Loan Size'][j]*IC
            tcost += rT*( (1-K)*D - sumloan )
            sumloss += df['Loan Loss'][j]
    
    costDD = rD*D

    fitness = (lrev + lcost + tcost - costDD - sumloss)
    return 1/fitness

#FUNCTION TO GENERATE INITIAL SOLUTION WITHIN FEASIBLE VALUES OF SUM OF LOAN AMOUNTS
def sa_generate_init_pop(df, N, K, D, rD, rT, IC, n=5):
    population = []
    fitness = []
    while(len(population) < n):
        rgs = [0]*N #randomly generated solution
        sumL = 0    #total loans approved
        for j in range(N):
            if random.random() >= 0.5:
                rgs[j] = 1
                sumL += df['Loan Size'][j]
            else:
                continue
        if sumL > (1 - K)*D:
            continue
        else:
            population.append(rgs)
            fitness.append(sa_fitness_eval(rgs, N, df, IC, K, D, rD, rT))
    init_temp = (sum(fitness)/len(fitness))*pow(10,10)
    return population[0], init_temp

#FUNCTION TO CHECK VALIDITY OF SOLUTION
def sa_check_validity(solution, K, D, df):
    N = len(solution)
    sumL = 0
    for i in range(N):
        if solution[i] == 1:
            sumL += df['Loan Size'][i]
    if sumL <= (1-K)*D:
        return 1
    else:
        return 0

#FUNCTION TO GENERATE NEIGHBORING SOLUTION OF INITIAL SOLUTION
def sa_generate_neighbor(solution, K, D, df):
    while(1):
        choice = random.randint(0, len(solution)-1 )
        neighbor = solution.copy()

        neighbor[choice] = int(not(neighbor[choice]))
        if sa_check_validity(neighbor, K, D, df) == 0:
            continue
        else:
            return neighbor

#MAIN FUNCTION TO IMPLEMENT SIMULATED ANNEALING
def sim_ann(df, D = 600000, K = 0.15, rD = 0.09, rT = 0.01, IC = 10, c=0.75, total_reductions=30, n_iter=3):
    
    N = len(df) #length of a solution

    sol_hist = []
    fit_hist = []
    new_solution, T_init = sa_generate_init_pop(df, N, K, D, rD, rT, IC)
    new_fit = sa_fitness_eval(new_solution, N, df, IC, K, D, rD, rT)
    sol_hist.append(new_solution)
    fit_hist.append(1/new_fit)
    T_fin = T_init*(pow(c, total_reductions))
    # print('T_init:', T_init, '\nT_fin:', T_fin)

    T_curr = T_init
    while(T_curr >= T_fin):
        for i in range(n_iter):
            old_solution = new_solution
            old_fit = new_fit
            
            poss_solution = sa_generate_neighbor(old_solution, K, D, df)
            poss_fit = sa_fitness_eval(poss_solution, N, df, IC, K, D, rD, rT)

            if poss_fit <= old_fit:
                new_solution = poss_solution
                new_fit = poss_fit
            else:
                probability = math.exp((old_fit - poss_fit)/T_curr)
                if random.random() <= probability:
                    new_solution = poss_solution
                    new_fit = poss_fit
                else:
                    new_solution = old_solution
                    new_fit = old_fit
            sol_hist.append(new_solution)
            fit_hist.append(1/new_fit)
        T_curr *= c
    results = pd.DataFrame({'SolutionHist':sol_hist, 'FitHist':fit_hist})

    return new_solution, new_fit, results