import math
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import cvxpy as cp
import time

def read_census(columns):
    data = pd.read_csv("data/USCensus1990.csv").head(1000000)
    data = data[columns]
    data["dAge"] = data["dAge"].apply(lambda x: 0 if x < 4 else 1)
    data = data[(data["dPoverty"] != 0)]
    data = pd.get_dummies(
        data,
        columns=columns,
        drop_first=True,
    )
    data["weight"] = np.random.randint(1, 280, size=len(data))
    return data


def run_dp(data, demands):
    n = len(data)
    dp = [
        [[[math.inf for _ in range(demands[2] + 5)] for _ in range(demands[1] + 5)] for _ in range(demands[0] + 5)]
        for _ in range(n + 5)
    ]
    dp[0][0][0][0] = 0
    for i in range(0, n):
        print(i, end="%\n")
        key = tuple(data.iloc[i][:-1])
        for j in range(demands[0] + 3):
            for k in range(demands[1] + 3):
                for l in range(demands[2] + 3):
                    w = data.iloc[i]["weight"]
                    dp[i + 1][j][k][l] = min(
                        dp[i][j][k][l], w + dp[i][max(0, j - key[0])][max(0, k - key[1])][max(0, l - key[2])]
                    )

    return dp[n][demands[0]][demands[1]][demands[2]]


def construct_priority_queues(data):
    dic = defaultdict(list)
    h = 0
    for _, row in data.iterrows():
        h += 1
        key = tuple(row[:-1])
        priority = row["weight"]
        dic[key].append(priority)
        if h % 50000 == 0:
            print(int(h / len(data) * 100), end="%\n")
    for key in dic.keys():
        dic[key].sort()
    return dic


def calculate_group_sum(key, demands):
    res = 0
    for i in range(len(key)):
        if key[i] > 0 and demands[i] > 0:
            res += 1
    return res


def find_best(dic, demands):
    all_keys = dic.keys()
    mn = math.inf
    best_key = -1
    for key in all_keys:
        if len(dic[key]) == 0:
            continue
        num_satisfy = calculate_group_sum(key, demands)
        curr = dic[key][0]
        if num_satisfy == 0:
            curr = math.inf
        else:
            curr /= num_satisfy
        if curr <= mn:
            mn = curr
            best_key = key
    if best_key == -1:
        return False
    return best_key


def is_finished(demands):
    res = True
    for demand in demands:
        if demand > 0:
            res = False
    return res


def run_greedy(dic, demands):
    res = 0
    result_list = []
    while not is_finished(demands):
        key = find_best(dic, demands)
        if not key:
            print("Demands Were Not Satisfied :(")
            return None
        value = dic[key][0]
        res += value
        result_list.append(dic[key][0])
        dic[key].pop(0)
        for i in range(len(key)):
            demands[i] -= key[i]
    return res


def run_lp(data, demands):
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


columns = ["iSex", "dPoverty", "dAge"]
data = read_census(columns)
res1_list = []
res2_list = []
res3_list = []
dp_construction_time = []
lp_construction_time = []
greedy_construction_time = []
for run in range(1):
    print("run: ", run)
    demands = [np.random.randint(1, 53) for col in columns]
    dic = construct_priority_queues(data)

    start_time = time.time()
    print(data)
    print(demands)
    res3 = run_lp(data, demands)
    end_time = time.time()
    lp_construction_time.append(end_time - start_time)
    res3_list.append(res3)
    
    # start_time = time.time()
    # res1 = run_dp(data, demands)
    # end_time = time.time()
    # dp_construction_time.append(end_time - start_time)
    # res1_list.append(res1)
    
    start_time = time.time()
    res2 = run_greedy(dic, demands)
    end_time = time.time()
    res2_list.append(res2)
    greedy_construction_time.append(end_time - start_time)
    print(res3_list)
    print(res2_list)

lp_construction_time_avg = np.mean(lp_construction_time)
# dp_construction_time_avg = np.mean(dp_construction_time)
greedy_construction_time_avg = np.mean(greedy_construction_time)

plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 20})
x = np.arange(len(res1_list))
width = 0.25


plt.bar(x - width, res1_list, width, label="DP", color="green")
plt.bar(x, res2_list, width, label="Greedy", color="orange")
plt.bar(x + width, res3_list, width, label="LP", color="blue")

plt.xticks(x, labels=range(1, len(res1_list) + 1))
plt.xlabel("Run")
plt.ylabel("Result Value")
plt.title("Comparison of DP, Greedy and LP Solutions")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.savefig("plot/comparison.png", bbox_inches="tight")

# plt.figure(figsize=(10, 6))
# plt.rcParams.update({"font.size": 20})
# approaches = ["LP", "DP", "Greedy"]
# construction_times = [lp_construction_time_avg, dp_construction_time_avg, greedy_construction_time_avg]

# plt.bar(approaches, construction_times, color=["blue", "green", "orange"])
# plt.xlabel("Approach")
# plt.ylabel("Average Construction Time (Sec)")
# plt.yscale("log")
# plt.title("Average Construction Time for LP, DP, and Greedy Approaches")
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.savefig("plot/time.png", bbox_inches="tight")
