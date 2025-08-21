from algorithms import run_greedy, run_lp, run_twist, construct_priority_queues, vizualize, rss_demands, visualize_rss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
plt.rcParams["font.family"] = "serif"


def read_synthetic(n, n_groups):
    data = pd.read_csv("data/synthetic.csv").head(n)
    data = data[data.columns[:n_groups]]
    weight = [1 / (n_groups - i) for i in range(n_groups)]
    weight.append(1.1)
    data["weight"] = weight
    return data


lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []

res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []


def build_demands(numgroups):
    l = []
    for i in range(numgroups):
        l.append(1)
    return l


lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []
lp_rss_avg = []
twist_rss_avg = []
greedy_rss_avg = []

num_groups = range(4, 64, 4)
for num_group in num_groups:
    n_size = num_group + 1
    print("n_groups: ", num_group)
    data = read_synthetic(n_size, num_group)
    res_greedy_list = []
    res_lp_list = []
    res_twist_list = []
    lp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []
    lp_rss = []
    twist_rss = []
    greedy_rss = []

    for i in range(30):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        dic2 = deepcopy(dic)
        print("PQ size:", len(dic))
        demands = build_demands(num_group)
        start_time = time.time()
        res_lp, covered_demands_lp = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)
        lp_rss.append(rss_demands(demands, covered_demands_lp))

        start_time = time.time()
        res_twist, covered_demands_twist = run_twist(dic2, demands)
        end_time = time.time()
        twist_construction_time.append(end_time - start_time)
        twist_rss.append(rss_demands(demands, covered_demands_twist))
        res_twist_list.append(res_twist)

        start_time = time.time()
        res_greedy, covered_demands_greedy = run_greedy(dic, demands)
        end_time = time.time()
        res_greedy_list.append(res_greedy)
        greedy_rss.append(rss_demands(demands, covered_demands_greedy))
        greedy_construction_time.append(end_time - start_time)

    lp_construction_time_avg.append(np.mean(lp_construction_time))
    greedy_construction_time_avg.append(np.mean(greedy_construction_time))
    twist_construction_time_avg.append(np.mean(twist_construction_time))

    res_lp_list_avg.append(np.mean(res_lp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))
    res_twist_list_avg.append(np.mean(res_twist_list))

    lp_rss_avg.append(np.mean(lp_rss))
    twist_rss_avg.append(np.mean(twist_rss))
    greedy_rss_avg.append(np.mean(greedy_rss))


vizualize(
    num_groups,
    res_lp_list_avg,
    res_greedy_list_avg,
    res_twist_list_avg,
    lp_construction_time_avg,
    greedy_construction_time_avg,
    twist_construction_time_avg,
    "Number of Groups",
    "Total Weight",
    "Time (sec)",
    "synthetic",
    "group",
)

# Plotting the RSS values
visualize_rss(
    num_groups,
    lp_rss_avg,
    greedy_rss_avg,
    twist_rss_avg,
    "synthetic"
)

num_group = 20
n_size = num_group + 1
data = read_synthetic(n_size, num_group)
demands = build_demands(num_group)
dic = construct_priority_queues(data)
res_lp, covered_lp = run_lp(data, demands)
res_twist, covered_twist = run_twist(dic, demands)
res_greedy, covered_greedy = run_greedy(dic, demands)
x = np.arange(num_group)

demands = np.array(demands[:num_group])
covered_lp = np.array(covered_lp[:num_group])
covered_twist = np.array(covered_twist[:num_group])
covered_greedy = np.array(covered_greedy[:num_group])

bar_width = 0.2

plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 20})
plt.bar(x - 1.5 * bar_width, demands, width=bar_width, label="Demand", color="green")
plt.bar(x + 0.5 * bar_width, covered_twist, width=bar_width, label="(2+ε)-approx.", color="red")
plt.bar(x + 1.5 * bar_width, covered_greedy, width=bar_width, label="Greedy", color="orange")
plt.bar(x - 0.5 * bar_width, covered_lp, width=bar_width, label="2-approx.", color="blue")
plt.xticks(x, [f"{i+1}" for i in range(num_group)])
plt.xlabel("Group")
plt.ylabel("Coverage")
plt.tight_layout()
plt.savefig("plot/synthetic_coverage_vs_demand.png", bbox_inches="tight")