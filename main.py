import pandas as pd
import numpy as np
from collections import defaultdict
import math


def read_census(columns):
    data = pd.read_csv("data/USCensus1990.csv")
    data = data[columns]
    data["dAge"] = data["dAge"].apply(lambda x: 0 if x < 4 else 1)
    data = data[(data["dPoverty"] != 0)]
    data = pd.get_dummies(
        data,
        columns=columns,
        drop_first=True,
    )
    data["weight"] = np.random.randint(0, 2000, size=len(data))
    return data


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
    return result_list


columns = [
    "iSex",
    "iRownchld",
    "dAge",
    "iFeb55",
    "iKorean",
    "iOthrserv",
    "dIncome2",
    "dIncome3",
    "dIncome4",
    "iVietnam",
    "dPoverty",
]

data = read_census(columns)
dic = construct_priority_queues(data)
column_demands = [np.random.randint(1000000, 100000000) for col in columns]
res = run_greedy(dic, column_demands)
if res is not None:
    print(len(res))
