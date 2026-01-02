import numpy as np
from pyscipopt import Model, Pricer, SCIP_RESULT, SCIP_PARAMSETTING, quicksum
import math
import random
import pandas as pd
import time

class SchedulePricer(Pricer):
    # The reduced cost function for the variable pricer
    def pricerredcost(self):

        # initialize parameters
        ub = 0
        tmp = 0
        currentNumVar = len(self.data['schedule_cols'])
        self.data['schedule_cols'][currentNumVar] = {}

        # Retrieving the dual solutions
        dualSolutions = [-self.model.getDualsolLinear(c) for i, c in enumerate(self.model.getConss())][self.data['nProcessors']:]

        # Define subproblems
        for i in range(1, self.data['nProcessors'] + 1):
            # Building a MIP to solve the subproblem
            subMIP = Model("rcpsp-Sub")
            subMIP.hideOutput(True)

            # Turning off presolve
            subMIP.setPresolve(SCIP_PARAMSETTING.OFF)

            # Setting the verbosity level to 0
            subMIP.hideOutput()

            subVars = []

            # Variables for the subMIP
            z = {}
            for j in self.data['jobs_dict'][i]:
                z[j] = {}
                for t in range(self.data['T']):
                    z[j][t] = subMIP.addVar(vtype="B", name="z(%s, %s)"%(j, t))
                    subVars.append(z[j][t])

            # Define Expressoins 1 and 2 for the objective Function
            expr1 = quicksum(self.data['cost'][j][t] * (z[j][t] - z[j][t-1]) for j in self.data['jobs_dict'][i] for t in range(self.data['T']) if t != 0)
            expr2 = quicksum(dualSolutions[t] * self.data['resource_usage'][j] * (z[j][t + self.data['process_time'][j]] - z[j][t]) for j in self.data['jobs_dict'][i] for t in range(self.data['T']) if t + self.data['process_time'][j] <  self.data['T'])

            # Set objective Function
            subMIP.setObjective(expr1 + expr2,'minimize')

            # Adding the constraints

            # Constraint 1: Jobs cannot be started before their release time
            for j in self.data['jobs_dict'][i]:
                for t in range(self.data['release_time'][j] + self.data['process_time'][j]):
                    subMIP.addCons(z[j][t] == 0)

            # Constraint 2: Each processor can process at most one job at a time
            for t in range(self.data['T']):
                subMIP.addCons(quicksum(z[j][t + self.data['process_time'][j]] - z[j][t] for j in self.data['jobs_dict'][i] if t + self.data['process_time'][j] < self.data['T']) <= 1)

            # Constraint 3: A job stays as completed once it is completed
            for j in self.data['jobs_dict'][i]:
                for t in range(self.data['T']):
                    if t != 0:
                        subMIP.addCons(z[j][t] >= z[j][t-1])

            # Constraint 4: All jobs must be completed at the end of the planning horizon, where T is the end of the time horizon
            for j in self.data['jobs_dict'][i]:
                subMIP.addCons(z[j][self.data['T'] - 1] == 1)

            # Constraint 5: Precedence relations must be satisfied
            for k in self.data['jobs_dict'][i]:
                for j in self.data['precedence'][k]:
                    for t in range(self.data['T']):
                        if t - self.data['process_time'][k] >= 0:
                            subMIP.addCons(z[k][t] <= z[j][t - self.data['process_time'][k]])

            # Constraint 6: The total resource consumed by all jobs at any one time should be no more than R_max
            for t in range(self.data['T']):
                subMIP.addCons(quicksum(self.data['resource_usage'][j] * (z[j][t + self.data['process_time'][j]] - z[j][t]) for j in self.data['jobs_dict'][i] if t + self.data['process_time'][j] < self.data['T']) <= self.data['R_max'])

            # Solving the subMIP to generate the most negative reduced cost pattern
            subMIP.optimize()

            # adding subproblem objective value to tmp for calculating lowerbound
            tmp += subMIP.getObjVal()

            # Print outputs
            print('--' * 25)
            print(f"Subproblem {i} at iteration {currentNumVar} \n Solution Status: {subMIP.getStatus()} \n Optimal value: {subMIP.getObjVal()}")

            # Adding the column to the master problem
            if self.model.getObjVal() -  self.data['LowerBound'] >= self.data['EPS']:

                # Update Schedule_cols set
                self.data['schedule_cols'][currentNumVar][i] = {}
                for j in self.data['jobs_dict'][i]:
                    self.data['schedule_cols'][currentNumVar][i][j] = []
                    for t in range(self.data['T']):
                        self.data['schedule_cols'][currentNumVar][i][j].append(subMIP.getVal(z[j][t]))

                # Update v and Q set
                for cols in self.data['schedule_cols'][currentNumVar].values():
                    self.data['v'][i][currentNumVar] = sum(self.data['cost'][j][t] * (cols[j][t] - cols[j][t-1]) for j in cols.keys() for t in range(self.data['T']) if t > 0)

                for cols in self.data['schedule_cols'][currentNumVar].values():
                    self.data['Q'][i][currentNumVar] = []
                    for t in range(self.data['T']):
                        self.data['Q'][i][currentNumVar].append(sum(self.data['resource_usage'][j] * (cols[j][t + self.data['process_time'][j]] - cols[j][t]) for j in cols.keys() if t + self.data['process_time'][j] < self.data['T']))

                # Creating new var; must set pricedVar to True
                newVar = self.model.addVar(vtype="C", name="x(%s, %s)"%(i, currentNumVar), obj = self.data['v'][i][currentNumVar], pricedVar = True)

                for idx, con in enumerate(self.model.getConss()):
                    if idx + 1 == i:
                        self.model.addConsCoeff(con, newVar, 1)

                # add new column to the master problem's resource constraints
                for t in range(self.data['T']):
                    for idx, con in enumerate(self.model.getConss()):
                        if idx == t + self.data['nProcessors']:
                            self.model.addConsCoeff(con, newVar, self.data['Q'][i][currentNumVar][t])

                self.data['var'].append(newVar)

        # get value of current variables on RMP and calculate ub based on heuristic 2
        sol = self.model.getBestSol()
        for i in range(len(self.data['var'])):
            solValue = self.model.getSolVal(sol, self.data['var'][i])
            if solValue > 0.5:
                ub += self.data['v'][int(str(self.data['var'][i])[2])][int(str(self.data['var'][i])[5])] * solValue

        # update tmp for calculating lowerbound at each iteration
        tmp -= sum(dualSolutions[t] * self.data['R_max'] for t in range(self.data['T']))

        # Update global LowerBound
        if tmp > self.data['LowerBound']:
            self.data['LowerBound'] = tmp

        # Update global UpperBound
        if self.data['LowerBound'] < ub < self.data['UpperBound']:
            self.data['UpperBound'] = ub

        print('--' * 25)
        print(f"Best Upper Bound = {self.data['UpperBound']} \nBest Lower Bound = {self.data['LowerBound']} \nRMP Objective Value at iteration {currentNumVar}: {self.model.getObjVal()}")

        print('**' * 25)
        return {'result':SCIP_RESULT.SUCCESS}

def cg_based_rcpsp():

    # Define schedule cols and parameters
    nProcessors = 3

    jobs_dict = {1: [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)],
                2: [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)],
                3:[(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]}

    jobs = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
        (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]

    R_max = 8

    resource_usage = {(1, 1): 3, (1, 2):4, (1, 3):2, (1, 4):2, (1, 5):4, (1, 6):3, (1, 7):4,
                    (2, 1):4, (2, 2):3, (2, 3):2, (2, 4):2, (2, 5):4, (2, 6):2, (2, 7):3, (2, 8):3,
                    (3, 1):3, (3, 2):3, (3, 3):2, (3, 4):2, (3, 5):4, (3, 6):3, (3, 7):3, (3, 8):3}

    T = 35

    process_time = {(1, 1): 5, (1, 2):4, (1, 3):6, (1, 4):6, (1, 5):3, (1, 6):1, (1, 7):4,
                    (2, 1):3, (2, 2):3, (2, 3):4, (2, 4):1, (2, 5):3, (2, 6):1, (2, 7):4, (2, 8):2,
                    (3, 1):3, (3, 2):6, (3, 3):1, (3, 4):4, (3, 5):3, (3, 6):3, (3, 7):3, (3, 8):2}

    weight_value = {(1, 1): 1, (1, 2):1, (1, 3):1, (1, 4):1, (1, 5):1, (1, 6):1, (1, 7):1,
                    (2, 1):1, (2, 2):1, (2, 3):1, (2, 4):1, (2, 5):1, (2, 6):1, (2, 7):1, (2, 8):1,
                    (3, 1):1, (3, 2):1, (3, 3):1, (3, 4):1, (3, 5):1, (3, 6):1, (3, 7):1, (3, 8):1}

    precedence =  {(1, 1): [], (1, 2): [], (1, 3): [], (1, 4): [(1, 1), (1, 2), (1, 3)], (1, 5): [(1, 3)], (1, 6): [(1, 1)], (1, 7): [(1, 4), (1, 5)],
                (2, 1): [], (2, 2): [], (2, 3): [], (2, 4): [(2, 1)], (2, 5): [(2, 1), (2, 2), (2, 3)], (2, 6): [(2, 3)], (2, 7): [(2, 4)], (2, 8): [(2, 5), (2, 6)],
                (3, 1): [], (3, 2): [], (3, 3): [(3, 1)], (3, 4): [(3, 1), (3, 2)], (3, 5): [(3, 2)], (3, 6): [(3, 3)], (3, 7): [(3, 4), (3, 5)], (3, 8): [(3, 5)]}

    release_time = {}
    for j in jobs:
        release_time[j] = sum([process_time[j_] for j_ in precedence[j]])


    due_time = {(1, 1): 6, (1, 2):11, (1, 3):18, (1, 4):25, (1, 5):27, (1, 6):29, (1, 7):34,
                (2, 1):4, (2, 2):8, (2, 3):13, (2, 4):15, (2, 5):19, (2, 6):21, (2, 7):26, (2, 8):29,
                (3, 1):4, (3, 2):11, (3, 3):13, (3, 4):18, (3, 5):22, (3, 6):26, (3, 7):30, (3, 8):33}

    cost = {}
    for j in jobs:
        cost[j] = {}
        for t in range(T):
            cost[j][t] = weight_value[j] * max((t - due_time[j]), 0)

    LowerBound = 0
    UpperBound = math.inf
    EPS = 1e-5

    schedule_cols = {0:{1: {(1, 1): [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (1, 2): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (1, 3): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (1, 4): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (1, 5): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (1, 6): [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (1, 7): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]},
                        2: {(2, 1): [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (2, 2): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (2, 3): [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (2, 4): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (2, 5): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (2, 6): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (2, 7): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (2, 8): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]},
                        3: {(3, 1): [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (3, 2): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (3, 3): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (3, 4): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (3, 5): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (3, 6): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        (3, 7): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        (3, 8): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]}}}

    v = {}
    for c in schedule_cols.keys():
        for i, cols in schedule_cols[c].items():
            v[i] = {}
            v[i][c] = sum(cost[j][t] * (cols[j][t] - cols[j][t-1]) for j in cols.keys() for t in range(T) if t > 0)

    Q = {}
    for c in schedule_cols.keys():
        for i, cols in schedule_cols[c].items():
            Q[i] = {}
            Q[i][c] = []
            for t in range(T):
                Q[i][c].append(sum(resource_usage[j] * (cols[j][t + process_time[j]] - cols[j][t]) for j in cols.keys() if t + process_time[j] < T))

    # create RMP
    master = Model("RCPSP Restricted Master Problem")
    master.hideOutput(True)

    # Turning off presolve
    master.setPresolve(0)

    # creating a pricer
    pricer = SchedulePricer()
    master.includePricer(pricer, "ResourceConstrainedSchedulingPricer",
                         "Pricer to identify new Scheduling solution")

    # adding the initial variables
    rcpspVars = []
    x = {}
    for c in schedule_cols.keys():
        for i, cols in schedule_cols[c].items():
            x[i] = {}
            x[i][c] = master.addVar(vtype="C", name="x(%s, %s)"%(i, c), obj = v[i][c])
            rcpspVars.append(x[i][c])

    # Adding constraints
    # Add constraint 1: Each processor can only select one column
    procceserCons = []
    for i in range(1, nProcessors + 1):
        procceserCons.append(master.addCons(quicksum(x[i][c] for c in schedule_cols.keys()) == 1,
                       separate = False, modifiable = True))

    # Add constraint 2: Resource constraint
    resourceCons = []
    for t in range(T):
        resourceCons.append(master.addCons(quicksum(Q[i][c][t] * x[i][c] for c in schedule_cols.keys() for i in schedule_cols[c].keys()) <= R_max,
                       separate = False, modifiable = True))

    # Setting the pricer_data for use in the init and redcost functions
    pricer.data = {}
    pricer.data['var'] = rcpspVars
    pricer.data['cons'] = resourceCons
    pricer.data['procceserCons'] = procceserCons
    pricer.data['v'] = v
    pricer.data['Q'] = Q
    pricer.data['schedule_cols'] = schedule_cols
    pricer.data['jobs_dict'] = jobs_dict
    pricer.data['nProcessors'] = nProcessors
    pricer.data['R_max'] = R_max
    pricer.data['resource_usage'] = resource_usage
    pricer.data['T'] = T
    pricer.data['process_time'] = process_time
    pricer.data['weight_value'] = weight_value
    pricer.data['precedence'] = precedence
    pricer.data['release_time'] = release_time
    pricer.data['due_time'] = due_time
    pricer.data['cost'] = cost
    pricer.data['LowerBound'] = LowerBound
    pricer.data['UpperBound'] = UpperBound
    pricer.data['EPS'] = EPS

    # solve problem
    master.optimize()
    solution = master.getBestSol()

    print('Optimal Objective Function Value:', master.getObjVal())

    print('======'*10)

    return master.getObjVal()

def alg_validation(cg_based_lb):

    # Number Of Proccesors
    nProcessors = 3

    # Ordering Jobs based on Processors
    jobs_dict = {1: [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)],
                2: [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)],
                3:[(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]}

    # List All Jobs
    jobs = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8),
            (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8)]

    # Maximum Available Resource at each day
    R_max = 8

    # Amount of resource usage by each job
    resource_usage = {(1, 1): 3, (1, 2):4, (1, 3):2, (1, 4):2, (1, 5):4, (1, 6):3, (1, 7):4,
                    (2, 1):4, (2, 2):3, (2, 3):2, (2, 4):2, (2, 5):4, (2, 6):2, (2, 7):3, (2, 8):3,
                    (3, 1):3, (3, 2):3, (3, 3):2, (3, 4):2, (3, 5):4, (3, 6):3, (3, 7):3, (3, 8):3}

    # Time Horizon
    T = 35

    # Process time of each job
    process_time = {(1, 1): 5, (1, 2):4, (1, 3):6, (1, 4):6, (1, 5):3, (1, 6):1, (1, 7):4,
                    (2, 1):3, (2, 2):3, (2, 3):4, (2, 4):1, (2, 5):3, (2, 6):1, (2, 7):4, (2, 8):2,
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

    # Create an ILP model
    model = Model('integer linear programmin model for RCPSP')
    model.hideOutput(True)

    # define decision variables
    z = {}
    for j in jobs:
        z[j] = {}
        for t in range(T):
            z[j][t] = model.addVar(vtype="C", name="z(%s, %s)"%(j, t))

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

    if cg_based_lb - model.getObjVal() < 1e-5:
        print("*** MODEL VALIDATION ***\n")
        print(f"Column Generation Based Lower Bound = {cg_based_lb} \nRelaxed ILP Lower Bound = {model.getObjVal()}")

if __name__ == '__main__':
    cg_based_lb = cg_based_rcpsp()
    alg_validation(cg_based_lb)