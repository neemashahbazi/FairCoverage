import math
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import heapq


def rss_demands(demand, cover):
    return sum((cover[i] - demand[i]) ** 2 for i in range(len(demand)))


def twist_rounding_schema(all_keys, vals, dmnds, funcs):
    demands = deepcopy(dmnds)
    all_sets = []
    costs = []
    cnt_covered_demands = [0 for _ in range(len(dmnds))]

    for i in range(len(all_keys)):
        key = all_keys[i]

        for j in range(vals[i] + 1, len(funcs[i])):
            cost = funcs[i][j] - funcs[i][j - 1]
            tea = deepcopy(key)
            tmp = tea + (cost,)
            all_sets.append(tmp)
            costs.append(cost)

        if vals[i] == 0:
            continue

        for j in range(len(demands)):
            if key[j] == 0:
                continue
            demands[j] -= vals[i]
            cnt_covered_demands[j] += vals[i]

    residual = list(demands)
    n = len(demands)
    m = len(all_sets)
    cover = [sum(1 for j in range(n) if all_sets[i][j] == 1 and residual[j] > 0) for i in range(m)]
    covering_sets = [[] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if all_sets[i][j] == 1:
                covering_sets[j].append(i)
    heap = []
    for i in range(m):
        if cover[i] > 0:
            ratio = all_sets[i][n] / cover[i]
            heapq.heappush(heap, (ratio, i, cover[i]))

    used = [0 for _ in range(m)]

    total_weight = 0
    while heap:
        ratio, i, cover_when_inserted = heapq.heappop(heap)
        if used[i]:
            continue
        if cover[i] == cover_when_inserted:
            total_weight += all_sets[i][n]
            used[i] = 1
            for j in range(n):
                if all_sets[i][j] == 1:
                    cnt_covered_demands[j] += 1
                if all_sets[i][j] == 1 and residual[j] > 0:
                    residual[j] -= 1
                    if residual[j] == 0:
                        for k in covering_sets[j]:
                            if cover[k] > 0:
                                cover[k] -= 1
                                if cover[k] > 0:
                                    new_ratio = all_sets[k][n] / cover[k]
                                    heapq.heappush(heap, (new_ratio, k, cover[k]))
    return total_weight, cnt_covered_demands


def lp_rounding_schema(all_keys, vals, dmnds, weights):
    demands = deepcopy(dmnds)
    all_sets = []
    costs = []
    cnt_covered_demands = [0 for _ in range(len(dmnds))]

    for i in range(len(all_keys)):
        key = all_keys[i]

        if vals[i] == 0:
            cost = weights[i]
            tea = deepcopy(key)
            tmp = tea + (cost,)
            all_sets.append(tmp)
            costs.append(cost)
            continue

        for j in range(len(demands)):
            if key[j] == 0:
                continue
            demands[j] -= vals[i]
            cnt_covered_demands[j] += vals[i]

    residual = list(demands)
    n = len(demands)
    m = len(all_sets)
    cover = [sum(1 for j in range(n) if all_sets[i][j] == 1 and residual[j] > 0) for i in range(m)]
    covering_sets = [[] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if all_sets[i][j] == 1:
                covering_sets[j].append(i)
    heap = []
    for i in range(m):
        if cover[i] > 0:
            ratio = all_sets[i][n] / cover[i]  # Weight / cover
            heapq.heappush(heap, (ratio, i, cover[i]))

    used = [0 for _ in range(m)]

    total_weight = 0
    while heap:
        ratio, i, cover_when_inserted = heapq.heappop(heap)
        if used[i]:
            continue
        if cover[i] == cover_when_inserted:
            total_weight += all_sets[i][n]
            used[i] = 1
            for j in range(n):
                if all_sets[i][j] == 1:
                    cnt_covered_demands[j] += 1
                if all_sets[i][j] == 1 and residual[j] > 0:
                    residual[j] -= 1
                    if residual[j] == 0:
                        for k in covering_sets[j]:
                            if cover[k] > 0:
                                cover[k] -= 1
                                if cover[k] > 0:
                                    new_ratio = all_sets[k][n] / cover[k]
                                    heapq.heappush(heap, (new_ratio, k, cover[k]))

    return total_weight, cnt_covered_demands


def run_greedy(dic, dmnds):
    demands = deepcopy(dmnds)
    all_keys = dic.keys()
    all_sets = []
    cnt_covered_demands = [0 for _ in range(len(dmnds))]
    for key in all_keys:
        costs = deepcopy(dic[key])
        for cost in costs:
            tea = deepcopy(key)
            tmp = tea + (cost,)
            all_sets.append(tmp)

    residual = list(demands)
    n = len(demands)
    m = len(all_sets)
    cover = [sum(1 for j in range(n) if all_sets[i][j] == 1 and residual[j] > 0) for i in range(m)]
    covering_sets = [[] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if all_sets[i][j] == 1:
                covering_sets[j].append(i)
    heap = []
    for i in range(m):
        if cover[i] > 0:
            ratio = all_sets[i][n] / cover[i]
            heapq.heappush(heap, (ratio, i, cover[i]))

    used = [0 for _ in range(m)]

    total_weight = 0
    while heap:
        ratio, i, cover_when_inserted = heapq.heappop(heap)
        if used[i]:
            continue
        if cover[i] == cover_when_inserted:
            total_weight += all_sets[i][n]
            used[i] = 1
            for j in range(n):
                if all_sets[i][j] == 1:
                    cnt_covered_demands[j] += 1
                if all_sets[i][j] == 1 and residual[j] > 0:
                    residual[j] -= 1
                    if residual[j] == 0:
                        for k in covering_sets[j]:
                            if cover[k] > 0:
                                cover[k] -= 1
                                if cover[k] > 0:
                                    new_ratio = all_sets[k][n] / cover[k]
                                    heapq.heappush(heap, (new_ratio, k, cover[k]))
    return total_weight, cnt_covered_demands


def calc_pwl(functions, demands, all_keys):
    max_demand = []
    for key in all_keys:
        max_tmp = 0
        for i in range(len(demands)):
            if key[i] == 0:
                continue
            max_tmp = max(max_tmp, demands[i])
        max_demand.append(max_tmp + 1)

    knots_x_list = []
    knots_y_list = []
    for i in range(len(functions)):
        func = functions[i]
        curr_dems = max_demand[i]
        curr_size = min(len(func), curr_dems + 1)
        curr_func = deepcopy(func[:curr_size])

        pt = 0

        curr_x = [0]
        curr_y = [0]
        eps = 2.0
        while pt < curr_size - 1:
            base_value = curr_func[pt]
            pt2 = pt + 1
            while pt2 < curr_size and curr_func[pt2] <= base_value * eps:
                pt2 += 1
            pt2 -= 1
            opt = max(pt + 1, pt2)
            pt = opt
            curr_x.append(pt)
            curr_y.append(curr_func[pt])

        knots_x_list.append(curr_x)
        knots_y_list.append(curr_y)

    return knots_x_list, knots_y_list


def run_twist(dic2, demands):
    all_keys = list(dic2.keys())
    convex_functions, cnt = [], []
    for key in all_keys:
        cur_q = dic2[key]
        ps, curr_f = 0, [0]
        for x in cur_q:
            ps += x
            curr_f.append(ps)
        convex_functions.append(curr_f)
        cnt.append(len(curr_f) - 1)

    knots_x_list, knots_y_list = calc_pwl(convex_functions, demands, all_keys)
    n, m = len(demands), len(cnt)

    model = gp.Model("Set_Multicover")
    x = [model.addVar(lb=0, ub=cnt[i], vtype=GRB.CONTINUOUS, name=f"x_{i}") for i in range(m)]
    z = [model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"z_{i}") for i in range(m)]
    model.setObjective(gp.quicksum(z), GRB.MINIMIZE)

    for i in range(m):
        knots_x, knots_y = knots_x_list[i], knots_y_list[i]
        for k in range(len(knots_x) - 1):
            x1, x2 = knots_x[k], knots_x[k + 1]
            y1, y2 = knots_y[k], knots_y[k + 1]
            slope, intercept = (y2 - y1) / (x2 - x1), y1 - ((y2 - y1) / (x2 - x1)) * x1
            model.addConstr(z[i] >= slope * x[i] + intercept, f"cost_{i}_{k}")

    for j in range(n):
        model.addConstr(gp.quicksum(x[i] * all_keys[i][j] for i in range(m)) >= demands[j], f"demand_{j}")

    model.setParam("OutputFlag", 0)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        print(f"Problem status: {model.Status}")
        return None, None

    solution = [var.X for var in x]
    optimal_cost = model.ObjVal
    print("Status: OPTIMAL")
    print(optimal_cost)

    optans, new_vals = 0, []
    for i, sol in enumerate(solution):
        val = int(round(sol))
        new_vals.append(val)
        optans += convex_functions[i][val]

    rounding_cost, cnt_covered_demands = twist_rounding_schema(all_keys, new_vals, demands, convex_functions)
    print("Rounding cost:", rounding_cost)
    return optans + rounding_cost, cnt_covered_demands


def run_lp(data, demands):
    n = len(data)
    L = [tuple(data.iloc[i]) for i in range(n)]
    A = np.array([row[:-1] for row in L])
    weights = np.array([row[-1] for row in L])

    model = gp.Model("Set_Cover")
    x = [model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_{i}") for i in range(n)]
    model.setObjective(gp.quicksum(weights[i] * x[i] for i in range(n)), GRB.MINIMIZE)

    for j in range(A.shape[1]):
        model.addConstr(gp.quicksum(A[i][j] * x[i] for i in range(n)) >= demands[j], f"demand_{j}")

    model.setParam("OutputFlag", 0)
    model.optimize()

    opt_value = model.ObjVal
    weighted_sum, new_vals = 0, []
    for i in range(n):
        val = int(round(x[i].X))
        weighted_sum += weights[i] * val
        new_vals.append(val)

    print("LP solution:")
    print(weighted_sum)
    print(opt_value)

    rounding_cost, cnt_covered_demands = lp_rounding_schema(L, new_vals, demands, weights)
    print("Rounding cost:", rounding_cost)
    return weighted_sum + rounding_cost, cnt_covered_demands


def construct_priority_queues(data):
    dic = defaultdict(list)
    h = 0
    for _, row in data.iterrows():
        h += 1
        key = tuple(row[:-1])
        priority = row["weight"]
        dic[key].append(priority)
    for key in dic.keys():
        dic[key].sort()
    return dic


def vizualize(x, y1, y2, y3, y4, y5, y6, x_lable, y_lable_1, y_lable_2, dataset, exp_type):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 20})
    width = 0.25
    plt.bar(np.arange(len(x)) + width, y1, width, label="2-approx.", color="blue")
    plt.bar(np.arange(len(x)), y2, width, label="Greedy", color="orange")
    plt.bar(np.arange(len(x)) - width, y3, width, label="(2+ϵ)-approx.", color="red")

    if exp_type == "n":
        plt.xticks(
            np.arange(len(x)),
            labels=[r"$2^{{{}}}$".format(i) for i in range(int(math.log2(x[0])), int(math.log2(x[-1])) + 1)],
        )
    elif exp_type == "distribution":
        plt.xticks(np.arange(len(x)), labels=[x[i][:4] + "." for i in range(len(x))])
    else:
        plt.xticks(np.arange(len(x)), x)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable_1)
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    # plt.legend()
    plt.savefig(f"plot/comparison_varying_{exp_type}_{dataset}.png", bbox_inches="tight")

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 20})
    plt.plot(x, y4, label="2-approx.", marker="o", color="blue")
    plt.plot(x, y5, label="Greedy", marker="^", color="orange")
    plt.plot(x, y6, label="(2+ϵ)-approx.", marker="x", color="red")

    if exp_type == "n":
        plt.xticks(
            np.arange(len(x)),
            labels=[r"$2^{{{}}}$".format(i) for i in range(int(math.log2(x[0])), int(math.log2(x[-1])) + 1)],
        )
        plt.xscale("log")
        plt.yscale("log")
    elif exp_type == "distribution":
        plt.xticks(np.arange(len(x)), labels=[x[i][:4] + "." for i in range(len(x))])
        plt.yscale("log")
        plt.xscale("linear")
    else:
        plt.xticks(np.arange(len(x)), x)
        plt.yscale("log")
        plt.xscale("linear")
    plt.xlabel(x_lable)
    plt.ylabel(y_lable_2)
    plt.grid(True)
    # plt.legend()
    plt.savefig(f"plot/time_varying_{exp_type}_{dataset}.png", bbox_inches="tight")


def visualize_rss(x, y1, y2, y3, dataset):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 20})
    plt.plot(x, y1, label="2-approx.", marker="o", color="blue")
    plt.plot(x, y2, label="Greedy", marker="^", color="orange")
    plt.plot(x, y3, label="(2+ϵ)-approx.", marker="x", color="red")
    plt.xlabel("Number of Groups")
    plt.ylabel("RSS")
    plt.grid(True)
    # plt.legend()
    plt.savefig(f"plot/rss_varying_group_{dataset}.png", bbox_inches="tight")
