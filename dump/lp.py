import cvxpy as cp
import pandas as pd
import numpy as np


def read_census(columns):
    data = pd.read_csv("data/USCensus1990.csv").head(10000)
    data = data[columns]
    data["dAge"] = data["dAge"].apply(lambda x: 0 if x < 4 else 1)
    data = data[(data["dPoverty"] != 0)]
    data = pd.get_dummies(
        data,
        columns=columns,
        drop_first=True,
    )
    data["weight"] = np.random.randint(0, 2000000, size=len(data))
    return data


def lp_sol(data,demands):
    n = len(data)
    L = []
    for i in range(n):
        L.append(tuple(data.iloc[i]))
    num_sets = len(L)
    A = np.array([row[:-1] for row in L])
    weights = np.array([row[-1] for row in L])
    x = cp.Variable(num_sets)
    constraints = [A.T @ x >= demands, x >= 0, x <= 1]
    objective = cp.Minimize(weights @ x)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    # print("Status:", prob.status)
    # print("Optimal value (total cost):", prob.value)
    # for i, val in enumerate(x.value):
    #     if val > 0.01:
    #         print(f"x[{i}] = {val:.4f}")
    return prob.value
