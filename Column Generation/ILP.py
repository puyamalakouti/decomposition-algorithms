import numpy as np
from pyscipopt import Model, quicksum
import pandas as pd

# Define paper Instances based on mentioned paper

# Number Of Proccesors
nProcessors = 3

# List of All Jobs
jobs = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]

# Ordering Jobs based on Processors
jobs_dict = {1: [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)],
             2: [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)],
             3:[(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]}

# Maximum Available Resource at each time
R_max = 8

# Amount of resource usage by each job
resource_usage = {(1, 1): 3, (1, 2):4, (1, 3):2, (1, 4):2, (1, 5):4, (1, 6):3, (1, 7):4,
                  (2, 1):4, (2, 2):3, (2, 3):2, (2, 4):2, (2, 5):4, (2, 6):2, (2, 7):3, (2, 8):3,
                  (3, 1):3, (3, 2):3, (3, 3):2, (3, 4):2, (3, 5):4, (3, 6):3, (3, 7):3, (3, 8):3}

# Time Horizon
T = 35

# Process time of each job
process_time = {(1, 1): 5, (1, 2):4, (1, 3):6, (1, 4):6, (1, 5):3, (1, 6):1, (1, 7):4,
                  (2, 1):3, (2, 2):3, (2, 3):3, (2, 4):1, (2, 5):3, (2, 6):1, (2, 7):4, (2, 8):2,
                  (3, 1):3, (3, 2):6, (3, 3):1, (3, 4):4, (3, 5):3, (3, 6):3, (3, 7):3, (3, 8):2}

# Precedence Relationship between jobs in processors
precedence =  {(1, 1): [], (1, 2): [], (1, 3): [], (1, 4): [(1, 1), (1, 2), (1, 3)], (1, 5): [(1, 3)], (1, 6): [(1, 1)], (1, 7): [(1, 4), (1, 5)],
               (2, 1): [], (2, 2): [], (2, 3): [], (2, 4): [(2, 1)], (2, 5): [(2, 1), (2, 2), (2, 3)], (2, 6): [(2, 3)], (2, 7): [(2, 4)], (2, 8): [(2, 5), (2, 6)],
               (3, 1): [], (3, 2): [], (3, 3): [(3, 1)], (3, 4): [(3, 1), (3, 2)], (3, 5): [(3, 2)], (3, 6): [(3, 3)], (3, 7): [(3, 4), (3, 5)], (3, 8): [(3, 5)]}

# Release time of each job
release_time = {}
for j in jobs:
    release_time[j] = sum([process_time[j_] for j_ in precedence[j]])

# due date of each job
due_time = {(1, 1): 6, (1, 2):11, (1, 3):18, (1, 4):25, (1, 5):27, (1, 6):29, (1, 7):34,
                  (2, 1):4, (2, 2):8, (2, 3):13, (2, 4):15, (2, 5):19, (2, 6):21, (2, 7):26, (2, 8):29,
                  (3, 1):4, (3, 2):11, (3, 3):13, (3, 4):18, (3, 5):22, (3, 6):26, (3, 7):30, (3, 8):33}

# Weight of each job
weight_value = {(1, 1): 1, (1, 2):1, (1, 3):1, (1, 4):1, (1, 5):1, (1, 6):1, (1, 7):1,
                  (2, 1):1, (2, 2):1, (2, 3):1, (2, 4):1, (2, 5):1, (2, 6):1, (2, 7):1, (2, 8):1,
                  (3, 1):1, (3, 2):1, (3, 3):1, (3, 4):1, (3, 5):1, (3, 6):1, (3, 7):1, (3, 8):1}

# Cost (Penalty of Tardiness) of each job
cost = {}
for j in jobs:
    cost[j] = {}
    for t in range(T):
        cost[j][t] = weight_value[j] * max((t - due_time[j]), 0)

# Modeling An ineteger Linear Programming (page 5)

# Create a model
model = Model('integer linear programming model for RCPSP')

# define decision variables
z = {}
for j in jobs:
    z[j] = {}
    for t in range(T):
        z[j][t] = model.addVar(vtype="B", name="z(%s, %s)"%(j, t))

# Objective Function
model.setObjective(quicksum(cost[j][t] * (z[j][t] - z[j][t-1]) for j in jobs for t in range(T) if t != 0), 'minimize')

# Constraint 1: Jobs cannot be started before their release time.
for j in jobs:
    for t in range(release_time[j] + process_time[j]):
        model.addCons(z[j][t] == 0)

# Constraint 2: Each processor can process at most one job at a time.
for i, j_i in jobs_dict.items():
    for t in range(T):
        model.addCons(quicksum(z[j][t + process_time[j]] - z[j][t] for j in j_i if t + process_time[j] < T) <= 1)

# Constraint 3: A job stays as completed once it is completed,
for j in jobs:
    for t in range(T):
        if t != 0:
            model.addCons(z[j][t] >= z[j][t-1])

# Constraint 4: All jobs must be completed at the end of the planning horizon, where T is the end of the time horizon.
for j in jobs:
    model.addCons(z[j][T - 1] == 1)

# Constraint 5: Precedence relations must be satisfied.
for k in jobs:
    for j in precedence[k]:
        for t in range(T):
            if t - process_time[k] >= 0:
                model.addCons(z[k][t] <= z[j][t - process_time[k]])

# Constraint 6: The total resource consumed by all jobs at any one time should be no more than R_max.
for t in range(T):
    model.addCons(quicksum(resource_usage[j] * (z[j][t + process_time[j]] - z[j][t]) for j in jobs if t + process_time[j] < T) <= R_max)

# Solve the problem
model.optimize()

# write the problem and print optimal objective function & decision variables
model.writeProblem('ILP.lp')

print(f"Status: {model.getStatus()}, Optimal value: {model.getObjVal()}")

# Extract outputs to excel file
tmp = {}
for j in jobs:
    tmp[j] = {}
    for t in range(T):
        tmp[j][t] = model.getVal(z[j][t])
df = pd.DataFrame(tmp).T
df.to_excel('output_ILP.xlsx', index=True)