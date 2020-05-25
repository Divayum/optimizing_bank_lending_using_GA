'''
====================================================================================================
Title:  OHM Term Project Code - Genetic Functions
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
OPTIMIZATION PARAMETERS FOR GAMCC (refer Table 5)
Population Size - n
Number of generations - gen
Crossover Ratio - crossover_ratio
Mutation Ratio - mutation_ratio
Reproduction Ratio - repro_ratio
****************************************************************************************************
RESULT VARIABLES FOR GAMCC
Individual population - population
Individual population fitness values - fit
All chromosomes ever generated and their fitness values - results
Generation average fitness values - avg_fitness
Generation maximum fitness values - max_fit
'''

#IMPORTING LIBRARIES
import numpy as np 
import pandas as pd 
import random

#FUNCTION TO GENERATE INITIAL POPULATION WITHIN FEASIBLE VALUES OF SUM OF LOAN AMOUNTS
def ga_generate_init_pop(df, N, n, K, D):
    population = []
    while(len(population) < n):
        rgc = [0]*N #randomly generated chromosomes
        sumL = 0    #total loans approved
        for j in range(N):
            if random.random() >= 0.5:
                rgc[j] = 1
                sumL += df['Loan Size'][j]
            else:
                continue
        if sumL > (1 - K)*D:
            continue
        else:
            population.append(rgc)
    return population

#FUNCTION TO EVALUATE FITNESS OF EACH SET OF CHROMOSOMES
def ga_fitness_eval(population, N, df, IC, K, D, rD, rT):
    fitness = []
    for i in population:
        lrev = 0    #nu
        lcost = 0   #mu
        tcost = 0   #omega
        sumloss = 0 #sumLambda
        costDD = 0  #beta
        sumloan = 0

        for j in range(N):
            if i[j] == 1:
                sumloan += df['Loan Size'][j]

        for j in range(N):
            if i[j] == 1:
                lrev += df['Interest Rate'][j]*df['Loan Size'][j] - df['Loan Loss'][j]
                lcost += df['Loan Size'][j]*IC
                tcost += rT*( (1-K)*D - sumloan )
                sumloss += df['Loan Loss'][j]
        
        costDD = rD*D

        fitness.append(lrev + lcost + tcost - costDD - sumloss)
    minimum = min(fitness)
    if minimum < 0:
        non_neg_fit = [x - minimum + 0.0001 for x in fitness]
    else:
        non_neg_fit = fitness.copy()
    return non_neg_fit

#FUNCTION TO CREATE A MATING POOL 
def ga_selection(population, fitness):
    fitsum = sum(fitness)
    probabilities = [x/fitsum for x in fitness]
    choices = np.random.choice(len(population), size = len(population), p = probabilities)
    new_population = []
    for i in choices:
        new_population.append(population[i])
    return new_population

#FUNCTION TO CHECK VALIDITY OF CHROMOSOMES
def ga_check_validity(chromosomes, K, D, df):
    N = len(chromosomes)
    sumL = 0
    for i in range(N):
        if chromosomes[i] == 1:
            sumL += df['Loan Size'][i]
    if sumL <= (1-K)*D:
        return 1
    else:
        return 0

#FUNCTION TO CARRY OUT CROSS-OVER IN MATING POOL
def ga_crossover_chromosomes(selected, crossover_ratio, K, D, df):
    n = len(selected)
    order = np.arange(0, n)
    random.shuffle(order)
    pairs = []
    for i in np.arange(0, n, step=2):
        if n%2 == 0:
            pairs.append([order[i], order[i+1]])
        elif i == n - 1:
            pairs.append([order[i]])
        else:
            pairs.append([order[i], order[i+1]])
    crossover = np.random.choice([0,1], len(pairs), p = [1-crossover_ratio, crossover_ratio])

    chromosome_pairs = []
    for i in pairs:
        pair = []
        for j in i:
            pair.append(selected[j])
        chromosome_pairs.append(pair)

    crossover_pairs = []
    for i in range(len(pairs)):
        if crossover[i] ==  1 and len(pairs[i]) == 2:
            position = random.randint(1, len(selected[0])-2)
            new_pair = [[0]*len(selected[0]),[0]*len(selected[0])]
            new_pair[0][:position] = chromosome_pairs[i][0][:position]
            new_pair[1][:position] = chromosome_pairs[i][1][:position]
            new_pair[0][position:] = chromosome_pairs[i][1][position:]
            new_pair[1][position:] = chromosome_pairs[i][0][position:]
            
            flag = 1
            for j in new_pair:
                if ga_check_validity(j, K, D, df) == 0:
                    flag = 0
            if flag == 0:
                new_pair = chromosome_pairs[i]
        else:
            new_pair = chromosome_pairs[i]
        crossover_pairs.append(new_pair)
    
    crossover_result = []
    for i in crossover_pairs:
        for j in i:
            crossover_result.append(j)
    return crossover_result

#FUNCTION TO CARRY OUT MUTATION OF INDIVIDUAL CHROMOSOMES
def ga_mutation(crossover_result, mutation_ratio, K, D, df):
    mutation_result = crossover_result.copy()
    n = len(mutation_result)
    mutation_yn = np.random.choice([0,1], n, p=[1-mutation_ratio, mutation_ratio])
    
    for i in range(n):
        if mutation_yn[i] == 1:
            position = random.randint(0, len(mutation_result[0])-1 )
            new_sol = mutation_result[i].copy()
            new_sol[position] = int(not(new_sol[position]))

            if ga_check_validity(new_sol, K, D, df) == 1:
                mutation_result[i] = new_sol
    return mutation_result

#MAIN FUNCTION TO IMPLEMENT GAMCC
def gamcc(df, D = 600000, K = 0.15, rD = 0.009, rT = 0.01, IC = 10, n = 60, gen = 60, crossover_ratio = 0.8, mutation_ratio = 0.006, repro_ratio = 0.194):
    N = len(df) #number of chromosomes

    init_population = ga_generate_init_pop(df, N, n, K, D)                  #generating initial population
    init_fit = ga_fitness_eval(init_population, N, df, IC, K, D, rD, rT)    #evaluating fitness
    avg_fit = []
    max_fit = []
    population_hist = []
    fit_hist = []

    population = init_population
    fit = init_fit
    avg_fit.append(sum(fit)/len(fit))
    max_fit.append(max(fit))
    population_hist.extend(population)
    fit_hist.extend(fit)

    for i in range(gen-1):
        selected = ga_selection(population, fit)
        crossover_result = ga_crossover_chromosomes(selected, crossover_ratio, K, D, df)
        mutation_result = ga_mutation(crossover_result, mutation_ratio, K, D, df)
        population = mutation_result
        fit = ga_fitness_eval(population, N, df, IC, K, D, rD, rT)
        avg_fit.append(sum(fit)/len(fit))
        max_fit.append(max(fit))
        population_hist.extend(population)
        fit_hist.extend(fit)
    
    results = pd.DataFrame({"PopulationHist":population_hist, "FitHist":fit_hist})
    return population, fit, results, avg_fit, max_fit