'''
====================================================================================================
Title:  OHM Term Project Code - Main Function
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
Individual population - gen_population
Individual population fitness values - gen_fit
All chromosomes ever generated and their fitness values - gen_results
Generation average fitness values - gen_avg_fitness
Generation maximum fitness values - gen_max_fit
****************************************************************************************************
OPTIMIZATION PARAMETERS FOR SIMULATED ANNEALING
Reduction factor - c
Total Number of Temperature Reductions - total_reductions
Number of iterations at one temperature - n_iters
****************************************************************************************************
RESULT VARIABLES FOR SIMULATED ANNEALING
Individual solution - simulation_solution
Solution fitness value - simulation_fit
All solutions ever generated and their fitness values - simulation_results
'''

#IMPORTING DATA HANDLING LIBRARIES
import numpy as np
import pandas as pd
import random

#IMPORTING VISUALISATION MODULES
import matplotlib.pyplot as plt
from matplotlib import style
style.use('Solarize_Light2')
get_ipython().run_line_magic('matplotlib', 'qt')

from tqdm import tqdm
import scipy.stats

#IMPORTING OPTIMISATION FUNCTIONS
from genetic_functions import *
from simulated_annealing_functions import *

#INPUTTING GIVEN DATA
data = [[10, 'AAA', 0.021, 0.0002],
        [25, 'BB', 0.022, 0.0058],
        [4, 'A', 0.021, 0.0001],
        [11, 'AA', 0.027, 0.0003],
        [18, 'BBB', 0.025, 0.0024],
        [3, 'AAA', 0.026, 0.0002],
        [17, 'BB', 0.023, 0.0058],
        [15, 'AAA', 0.021, 0.0002],
        [9, 'A', 0.028, 0.001],
        [10, 'A', 0.022, 0.001]]
df = pd.DataFrame(data, columns=['Loan Age', 'Credit Rating', 'Interest Rate', 'Loan Loss'])    #converting into a Pandas Dataframe

# L = []
# for i in range(len(df)):
#         L.append(round(50000 + random.random()*49999),2)

L = [79959.32, 64334.01, 90327.01, 94408.92, 84621.19, 71544.0, 90366.16, 81913.15, 67991.48, 55615.93] #randomly generated values of loan size considering loan to be of medium category (explained in report)
df.insert(1, "Loan Size", L, True)

ltype = []      #determining Loan Type on the basis of Table 3
for i in df['Interest Rate']:
        if (i >= 0.021 and i <= 0.028) or i == 0:
                ltype.append('M')
        elif i >= 0.0339 and i <= 0.0399:
                ltype.append('A')
        elif i >= 0.0599 and i <= 0.0609:
                ltype.append('P')
        else:
                ltype.append('')
df.insert(2, "Loan Type", ltype, True)
df

gen_population, gen_fit, gen_results, gen_avg_fit, gen_max_fit = gamcc(df)
maxvalindex = gen_results['FitHist'].idxmax(axis = 0)
print(gen_results['PopulationHist'][maxvalindex], gen_results['FitHist'][maxvalindex])

font = {'family':'calibri',
        'color':'grey',
        'weight':'bold',
        'size':20}
plt.plot(gen_avg_fit, label="Average Fitness per generation")
plt.plot(gen_max_fit, 'r--', label="Maximum Fitness per generation")
plt.legend()
plt.title("Fitness vs Generations", fontdict=font)
plt.xlabel("Generations")
plt.ylabel("Fitness Value")
plt.show()

gen_results.to_csv('GAMCC/GAMCC_Results.csv', index=False)

simulation_solution, simulation_fit, simulation_results = sim_ann(df)
maxvalindex = simulation_results['FitHist'].idxmax(axis = 0)
print(simulation_results['SolutionHist'][maxvalindex], simulation_results['FitHist'][maxvalindex])

font = {'family':'calibri',
        'color':'grey',
        'weight':'bold',
        'size':20}
plt.plot(simulation_results['FitHist'], label="Fitness for each solution")
plt.legend()
plt.title("Fitness vs Solution numbers", fontdict=font)
plt.xlabel("Solutions")
plt.ylabel("Fitness Value")
plt.show()

simulation_results.to_csv('Simulated Annealing/SimulatedAnnealing_Results.csv', index=False)

'''
====================================================================================================
ANALYZING THE RESULTS FOR COMPARISION BETWEEN GENETIC ALGORITHM AND SIMULATED ANNEALING
====================================================================================================
'''

iterations = 100

gen_max_fitness = []
for i in tqdm(range(iterations)):
    gen_population, gen_fit, gen_results, gen_avg_fit, gen_max_fit = gamcc(df)
    gen_max_fitness.append(max(gen_results['FitHist']))

simulation_max_fitness = []
for i in tqdm(range(iterations)):
    simulation_solution, simulation_fit, simulation_results = sim_ann(df)
    simulation_max_fitness.append(max(simulation_results['FitHist']))

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

gen_m, gen_low, gen_high = mean_confidence_interval(gen_max_fitness)
sim_m, sim_low, sim_high = mean_confidence_interval(simulation_max_fitness)

print('Confidence interval for Genetic Algorithm maximum results is:', gen_low, 'to', gen_high)
print('Confidence interval for Simulated Annealing maximum results is:', sim_low, 'to', sim_high)