import math
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import cvxpy as cp
import time
from copy import deepcopy


def read_census(columns, n):
    data = pd.read_csv("data/USCensus1990.csv").head(n)
    data = data[columns]
    for col in columns:
        median_value = data[col].mean()
        data[col] = data[col].apply(lambda x: 0 if x < median_value else 1)
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
        # print(i, end="%\n")
        key = tuple(data.iloc[i][:-1])
        for j in range(demands[0] + 3):
            for k in range(demands[1] + 3):
                for l in range(demands[2] + 3):
                    w = data.iloc[i]["weight"]
                    dp[i + 1][j][k][l] = min(
                        dp[i][j][k][l], w + dp[i][max(0, j - key[0])][max(0, k - key[1])][max(0, l - key[2])]
                    )

    return dp[n][demands[0]][demands[1]][demands[2]]


def run_greedy(dic, demands):
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

    res = 0
    result_list = []
    while not is_finished(demands):
        key = find_best(dic, demands)
        if not key:
            # print("Demands Were Not Satisfied :(")
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
    return prob.value


def construct_priority_queues(data):
    dic = defaultdict(list)
    h = 0
    for _, row in data.iterrows():
        h += 1
        key = tuple(row[:-1])
        priority = row["weight"]
        dic[key].append(priority)
        # if h % 50000 == 0:
        # print(int(h / len(data) * 100), end="%\n")
    for key in dic.keys():
        dic[key].sort()
    return dic


columns = ["iSex", "dPoverty", "dAge"]
lp_construction_time_avg = []
dp_construction_time_avg = []
greedy_construction_time_avg = []
res_dp_list_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
demands_org = [np.random.randint(1, 53) for col in columns]
range_n = range(10, 22)
n_values = [2**i for i in range_n]
for n in n_values:
    print("n: ", n)
    data = read_census(columns, n)
    res_dp_list = []
    res_greedy_list = []
    res_lp_list = []
    dp_construction_time = []
    lp_construction_time = []
    greedy_construction_time = []
    for i in range(1):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        demands = deepcopy(demands_org)
        start_time = time.time()
        res_lp = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)

        # start_time = time.time()
        # res_dp = run_dp(data, demands)
        # end_time = time.time()
        # dp_construction_time.append(end_time - start_time)
        # res_dp_list.append(res_dp)

        start_time = time.time()
        res_greedy = run_greedy(dic, demands)
        end_time = time.time()
        res_greedy_list.append(res_greedy)
        greedy_construction_time.append(end_time - start_time)

    lp_construction_time_avg.append(np.mean(lp_construction_time))
    dp_construction_time_avg.append(np.mean(dp_construction_time))
    greedy_construction_time_avg.append(np.mean(greedy_construction_time))

    res_lp_list_avg.append(np.mean(res_lp_list))
    res_dp_list_avg.append(np.mean(res_dp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))

plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 20})
plt.plot(n_values, lp_construction_time_avg, label="LP", marker="o")
# plt.plot(n_values, dp_construction_time_avg, label="DP", marker="s")
plt.plot(n_values, greedy_construction_time_avg, label="Greedy", marker="^")
plt.xticks(n_values, labels=[r"$2^{{{}}}$".format(i) for i in range_n])
plt.xlabel("Number of Records")
plt.ylabel("Average Construction Time (sec)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid(True)
plt.savefig("plot/time_varying_n.png", bbox_inches="tight")


plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 20})
x = np.arange(len(n_values))
width = 0.25
# plt.bar(x - width, res_dp_list_avg, width, label="DP", color="green")
plt.bar(x, res_greedy_list_avg, width, label="Greedy", color="orange")
plt.bar(x + width, res_lp_list_avg, width, label="LP", color="blue")
plt.xticks(x, labels=[r"$2^{{{}}}$".format(i) for i in range_n])
plt.xlabel("Number of Records")
plt.ylabel("Total Weight")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.savefig("plot/comparison_varying_n.png", bbox_inches="tight")


all_columns = [
    "iSex",
    "dPoverty",
    "dAge",
    "iFeb55",
    "dIncome2",
    "dIncome3",
    "dIncome4",
    "dIncome5",
    "dIncome6",
    "dIncome7",
    "dIncome8",
    "iKorean",
    "iMay75880",
    "iRownchld",
    "iRrelchld",
    "iSept80",
    "iVietnam",
    "iWWII",
    "dRearning",
    "iOthrserv",
    "dIncome1",
    "dAncstry1",
    "dAncstry2",
    "iAvail",
    "iCitizen",
    "iClass",
    "dDepart",
    "iDisabl1",
    "iDisabl2",
    "iEnglish",
    "iFertil",
    "dHispanic",
    "dHour89",
    "dHours",
    "iImmigr",
    "dIndustry",
    "iLang1",
    "iLooking",
    "iMarital",
    "iMeans",
    "iMilitary",
    "iMobility",
    "iMobillim",
    "dOccup",
    "iPerscare",
    "dPOB",
    "dPwgt1",
    "iRagechld",
    "iRelat1",
    "iRelat2",
    "iRemplpar",
    "iRiders",
    "iRlabor",
    "dRpincome",
    "iRPOB",
    "iRspouse",
    "iRvetserv",
    "iSchool",
    "iSubfam1",
    "iSubfam2",
    "iTmpabsnt",
    "dTravtime",
    "dWeek89",
    "iWork89",
    "iWorklwk",
    "iYearsch",
    "iYearwrk",
    "dYrsserv",
]
lp_construction_time_avg = []
dp_construction_time_avg = []
greedy_construction_time_avg = []
res_dp_list_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
demands_org = [np.random.randint(1, 53) for col in all_columns]
num_groups = range(3, 69, 5)
for num_group in num_groups:
    columns = all_columns[0:num_group]
    print("n_groups: ", num_group)
    data = read_census(columns, 50000)
    res_dp_list = []
    res_greedy_list = []
    res_lp_list = []
    dp_construction_time = []
    lp_construction_time = []
    greedy_construction_time = []
    for i in range(1):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        print("PQ size:", len(dic))
        demands = deepcopy(demands_org[:num_group])
        start_time = time.time()
        res_lp = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)

        # start_time = time.time()
        # res_dp = run_dp(data, demands)
        # end_time = time.time()
        # dp_construction_time.append(end_time - start_time)
        # res_dp_list.append(res_dp)

        start_time = time.time()
        res_greedy = run_greedy(dic, demands)
        end_time = time.time()
        res_greedy_list.append(res_greedy)
        greedy_construction_time.append(end_time - start_time)

    lp_construction_time_avg.append(np.mean(lp_construction_time))
    dp_construction_time_avg.append(np.mean(dp_construction_time))
    greedy_construction_time_avg.append(np.mean(greedy_construction_time))

    res_lp_list_avg.append(np.mean(res_lp_list))
    res_dp_list_avg.append(np.mean(res_dp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))

plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 20})
plt.plot(num_groups, lp_construction_time_avg, label="LP", marker="o")
# plt.plot(n_values, dp_construction_time_avg, label="DP", marker="s")
plt.plot(num_groups, greedy_construction_time_avg, label="Greedy", marker="^")
plt.xticks(num_groups)
plt.xlabel("Number of Groups")
plt.ylabel("Average Construction Time (sec)")
plt.legend()
plt.grid(True)
plt.savefig("plot/time_varying_group.png", bbox_inches="tight")


plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 20})
x = np.arange(len(num_groups))
width = 0.25
# plt.bar(x - width, res_dp_list_avg, width, label="DP", color="green")
plt.bar(x, res_greedy_list_avg, width, label="Greedy", color="orange")
plt.bar(x + width, res_lp_list_avg, width, label="LP", color="blue")
plt.xticks(x, num_groups)
plt.xlabel("Number of Groups")
plt.ylabel("Total Weight")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.savefig("plot/comparison_varying_group.png", bbox_inches="tight")
