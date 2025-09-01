from algorithms import run_greedy, run_lp, run_twist, construct_priority_queues, vizualize, visualize_rss, rss_demands,run_lp_randomized_rounding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

plt.rcParams["font.family"] = "serif"


def read_census(columns, n):
    np.random.seed(42)
    data = pd.read_csv("data/USCensus1990.csv").head(n)
    data = data[columns]
    for col in columns:
        median_value = data[col].mean()
        data[col] = data[col].apply(lambda x: 0 if x < median_value else 1)
    data["weight"] = np.random.randint(1, 1000, size=len(data))
    return data


def build_demands(dic, numgroups, n, distribution="random"):
    np.random.seed(42)
    cnt_sets = [0 for _ in range(numgroups)]
    all_keys = dic.keys()
    cnt = 0
    for key in all_keys:
        cnt_key = len(dic[key])
        cnt += cnt_key
        for j in range(len(key)):
            cnt_sets[j] += cnt_key * key[j]
    l = []
    for i in range(numgroups):
        if distribution == "random":
            tmp = np.random.randint(1, 100)
        elif distribution == "uniform":
            tmp = n // 3000 + 2
        elif distribution == "normal":
            tmp = int(np.random.normal(n // 6000 + 2, 1))
        elif distribution == "exponential":
            tmp = int(np.random.exponential(n // 6000 + 2))
        elif distribution == "poisson":
            tmp = int(np.random.poisson(n // 6000 + 2))
        elif distribution == "zipf":
            tmp = int(np.random.zipf(2))
        mn = cnt_sets[i]
        l.append(min(tmp, mn))
    return l


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
greedy_construction_time_avg = []
twist_construction_time_avg = []
lp_randomized_construction_time_avg = []

res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []
res_lp_randomized_list_avg = []

range_n = range(10, 21)
n_values = [2**i for i in range_n]
for n in n_values:
    print("n: ", n)
    group_cnt = 21
    columns = all_columns[0:group_cnt]
    data = read_census(columns, n)
    res_greedy_list = []
    res_lp_list = []
    res_twist_list = []
    res_lp_randomized_list = []

    lp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []
    lp_randomized_construction_time = []
    for i in range(5):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        dic2 = deepcopy(dic)
        demands = build_demands(dic2, group_cnt, n)
        start_time = time.time()
        res_lp, _ = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)

        start_time = time.time()
        res_lp_randomized, _ = run_lp_randomized_rounding(data, demands)
        end_time = time.time()
        lp_randomized_construction_time.append(end_time - start_time)
        res_lp_randomized_list.append(res_lp_randomized)

        start_time = time.time()
        res_twist, _ = run_twist(dic2, demands)
        end_time = time.time()
        twist_construction_time.append(end_time - start_time)
        res_twist_list.append(res_twist)

        start_time = time.time()
        res_greedy, _ = run_greedy(dic, demands)
        end_time = time.time()
        res_greedy_list.append(res_greedy)
        greedy_construction_time.append(end_time - start_time)

    lp_construction_time_avg.append(np.mean(lp_construction_time))
    greedy_construction_time_avg.append(np.mean(greedy_construction_time))
    twist_construction_time_avg.append(np.mean(twist_construction_time))
    lp_randomized_construction_time_avg.append(np.mean(lp_randomized_construction_time))

    res_lp_list_avg.append(np.mean(res_lp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))
    res_twist_list_avg.append(np.mean(res_twist_list))
    res_lp_randomized_list_avg.append(np.mean(res_lp_randomized_list))

vizualize(
    n_values,
    res_lp_list_avg,
    res_greedy_list_avg,
    res_twist_list_avg,
    res_lp_randomized_list_avg,
    lp_construction_time_avg,
    greedy_construction_time_avg,
    twist_construction_time_avg,
    lp_randomized_construction_time_avg,
    "Number of Sets",
    "Total Weight",
    "Time (sec)",
    "census",
    "n",
)

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
lp_randomized_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []
res_lp_randomized_list_avg = []
lp_rss_avg = []
twist_rss_avg = []
greedy_rss_avg = []
lp_randomized_rss_avg = []

num_groups = range(3, 68, 5)
n_size = 100000
for num_group in num_groups:
    columns = all_columns[0:num_group]
    print("n_groups: ", num_group)
    data = read_census(columns, n_size)
    res_greedy_list = []
    res_lp_list = []
    res_twist_list = []
    res_lp_randomized_list = []
    lp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []
    lp_randomized_construction_time = []
    lp_rss = []
    twist_rss = []
    greedy_rss = []
    lp_randomized_rss = []

    for i in range(5):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        dic2 = deepcopy(dic)
        print("PQ size:", len(dic))
        demands = build_demands(dic2, num_group, n_size)
        start_time = time.time()
        res_lp, covered_demands_lp = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)
        lp_rss.append(rss_demands(demands, covered_demands_lp))
        
        start_time = time.time()
        res_lp_randomized, covered_demands_lp_randomized = run_lp_randomized_rounding(data, demands)
        end_time = time.time()
        lp_randomized_construction_time.append(end_time - start_time)
        res_lp_randomized_list.append(res_lp_randomized)
        lp_randomized_rss.append(rss_demands(demands, covered_demands_lp_randomized))

        start_time = time.time()
        res_twist, covered_demands_twist = run_twist(dic2, demands)
        end_time = time.time()
        twist_construction_time.append(end_time - start_time)
        res_twist_list.append(res_twist)
        twist_rss.append(rss_demands(demands, covered_demands_twist))

        start_time = time.time()
        res_greedy, covered_demands_greedy = run_greedy(dic, demands)
        end_time = time.time()
        res_greedy_list.append(res_greedy)
        greedy_construction_time.append(end_time - start_time)
        greedy_rss.append(rss_demands(demands, covered_demands_greedy))

    lp_construction_time_avg.append(np.mean(lp_construction_time))
    greedy_construction_time_avg.append(np.mean(greedy_construction_time))
    twist_construction_time_avg.append(np.mean(twist_construction_time))
    lp_randomized_construction_time_avg.append(np.mean(lp_randomized_construction_time))
    lp_rss_avg.append(np.mean(lp_rss))
    twist_rss_avg.append(np.mean(twist_rss))
    greedy_rss_avg.append(np.mean(greedy_rss))
    lp_randomized_rss_avg.append(np.mean(lp_randomized_rss))

    res_lp_list_avg.append(np.mean(res_lp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))
    res_twist_list_avg.append(np.mean(res_twist_list))
    res_lp_randomized_list_avg.append(np.mean(res_lp_randomized_list))


vizualize(
    num_groups,
    res_lp_list_avg,
    res_greedy_list_avg,
    res_twist_list_avg,
    res_lp_randomized_list_avg,
    lp_construction_time_avg,
    greedy_construction_time_avg,
    twist_construction_time_avg,
    lp_randomized_construction_time_avg,
    "Number of Items",
    "Total Weight",
    "Time (sec)",
    "census",
    "group",
)

visualize_rss(
    num_groups,
    lp_rss_avg,
    greedy_rss_avg,
    twist_rss_avg,
    lp_randomized_rss_avg,
    "census",
)

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
lp_randomized_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []
res_lp_randomized_list_avg = []

num_group = 21
n_size = 100000
distributions = ["random", "uniform", "normal", "exponential", "poisson", "zipf"]
for dist in distributions:
    columns = all_columns[:num_group]
    print("distributions: ", dist)
    data = read_census(columns, n_size)
    res_greedy_list = []
    res_lp_list = []
    res_twist_list = []
    res_lp_randomized_list = []
    lp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []
    lp_randomized_construction_time = []
    
    for i in range(5):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        dic2 = deepcopy(dic)
        print("PQ size:", len(dic))
        demands = build_demands(dic2, num_group, n_size, dist)
        start_time = time.time()
        res_lp, _ = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)

        start_time = time.time()
        res_lp_randomized, _ = run_lp_randomized_rounding(data, demands)
        end_time = time.time()
        lp_randomized_construction_time.append(end_time - start_time)
        res_lp_randomized_list.append(res_lp_randomized)

        start_time = time.time()
        res_twist, _ = run_twist(dic2, demands)
        end_time = time.time()
        twist_construction_time.append(end_time - start_time)
        res_twist_list.append(res_twist)

        start_time = time.time()
        res_greedy, _ = run_greedy(dic, demands)
        end_time = time.time()
        res_greedy_list.append(res_greedy)
        greedy_construction_time.append(end_time - start_time)

    lp_construction_time_avg.append(np.mean(lp_construction_time))
    greedy_construction_time_avg.append(np.mean(greedy_construction_time))
    twist_construction_time_avg.append(np.mean(twist_construction_time))
    lp_randomized_construction_time_avg.append(np.mean(lp_randomized_construction_time))

    res_lp_list_avg.append(np.mean(res_lp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))
    res_twist_list_avg.append(np.mean(res_twist_list))
    res_lp_randomized_list_avg.append(np.mean(res_lp_randomized_list))

vizualize(
    distributions,
    res_lp_list_avg,
    res_greedy_list_avg,
    res_twist_list_avg,
    res_lp_randomized_list_avg,
    lp_construction_time_avg,
    greedy_construction_time_avg,
    twist_construction_time_avg,
    lp_randomized_construction_time_avg,
    "Distribution of Demands",
    "Total Weight",
    "Time (sec)",
    "census",
    "distribution",
)
