import math
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import cvxpy as cp
import time
from copy import deepcopy
import warnings
import pulp
import heapq

def read_census(columns, n):
    data = pd.read_csv("data/USCensus1990.csv").head(n)
    data = data[columns]
    for col in columns:
        median_value = data[col].mean()
        data[col] = data[col].apply(lambda x: 0 if x < median_value else 1)
    data["weight"] = np.random.randint(1, 28000000, size=len(data))
    return data


def run_dp(data, demands):
    n = len(data)
    dp = [
        [[[math.inf for _ in range(demands[2] + 5)] for _ in range(demands[1] + 5)] for _ in range(demands[0] + 5)]
        for _ in range(n + 5)
    ]
    dp[0][0][0][0] = 0
    for i in range(0, n):
        key = tuple(data.iloc[i][:-1])
        for j in range(demands[0] + 3):
            for k in range(demands[1] + 3):
                for l in range(demands[2] + 3):
                    w = data.iloc[i]["weight"]
                    dp[i + 1][j][k][l] = min(
                        dp[i][j][k][l], w + dp[i][max(0, j - key[0])][max(0, k - key[1])][max(0, l - key[2])]
                    )

    return dp[n][demands[0]][demands[1]][demands[2]]


def run_dynamic_greedy(dic, dmnds):
    demands = deepcopy(dmnds)
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
            return None
        value = dic[key][0]
        res += value
        result_list.append(dic[key][0])
        dic[key].pop(0)
        for i in range(len(key)):
            demands[i] -= key[i]
    return res


def run_greedy(dic, dmnds):
    demands = deepcopy(dmnds)
    all_keys = dic.keys()
    all_sets = []
    for key in all_keys:
        costs = deepcopy(dic[key])
        for cost in costs:
            tea = deepcopy(key)
            tmp = tea + (cost, )
            all_sets.append(tmp)
            
    residual = list(demands)
    n = len(demands)
    m = len(all_sets)
    
    # Initialize cover for each set: number of elements with residual > 0 it covers
    cover = [sum(1 for j in range(n) if all_sets[i][j] == 1 and residual[j] > 0) for i in range(m)]
    
    # Build covering_sets: for each element j, list of sets i that cover it
    covering_sets = [[] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if all_sets[i][j] == 1:
                covering_sets[j].append(i)
    
    # Initialize priority queue with (ratio, set index, cover when inserted)
    heap = []
    for i in range(m):
        if cover[i] > 0:
            ratio = all_sets[i][n] / cover[i]  # Weight / cover
            heapq.heappush(heap, (ratio, i, cover[i]))
            
    used = [0 for _ in range(m)]
    
    total_weight = 0
    
    # Process until all demands are satisfied (heap empties when all cover[i] = 0)
    while heap:
        ratio, i, cover_when_inserted = heapq.heappop(heap)
        if used[i]:
            continue 
        if cover[i] == cover_when_inserted:
            # Valid set: select it
            total_weight += all_sets[i][n]
            used[i] = 1
            # Update residuals and covers
            for j in range(n):
                if all_sets[i][j] == 1 and residual[j] > 0:
                    residual[j] -= 1
                    if residual[j] == 0:
                        # Element j is now fully covered, update all sets covering j
                        for k in covering_sets[j]:
                            if cover[k] > 0:
                                cover[k] -= 1
                                if cover[k] > 0:
                                    new_ratio = all_sets[k][n] / cover[k]
                                    heapq.heappush(heap, (new_ratio, k, cover[k]))
        
    
    return total_weight 


def run_cp(dic2, demands):
    all_keys = list(dic2.keys())
    convex_functions = []
    cnt = []
    for key in all_keys:
        cur_q = dic2[key]
        ps = 0
        curr_f = [0]
        for x in cur_q:
            ps += x
            curr_f.append(ps)
        convex_functions.append(curr_f)
        cnt.append(len(curr_f) - 1)

    n = len(demands)
    m = len(all_keys)

    x = cp.Variable(m)
    cost_vars = []
    constraints = []

    for j in range(m):
        k_max = cnt[j]
        lambdas = cp.Variable(k_max + 1, nonneg=True)
        k_vals = np.arange(k_max + 1)
        f_vals = np.array(convex_functions[j])

        constraints.append(x[j] == k_vals @ lambdas)
        constraints.append(cp.sum(lambdas) == 1)
        cost = f_vals @ lambdas
        cost_vars.append(cost)

    for i in range(n):
        coverage = sum(x[j] * all_keys[j][i] for j in range(m))
        constraints.append(coverage >= demands[i])

    total_cost = cp.sum(cost_vars)
    problem = cp.Problem(cp.Minimize(total_cost), constraints)
    problem.solve()

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise RuntimeError("Solver failed.")

    return problem.value


def fit_quadratic(f_list):
    """
    Fits a quadratic polynomial to each piecewise linear array in f_list.
    Returns a list of coefficient tuples (a, b, c) for each g_i(x) = ax^2 + bx + c.
    """
    coeffs = []
    for f in f_list:
        x = np.arange(len(f))
        y = np.array(f)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                a, b, c = np.polyfit(x, y, deg=2)
                tt = a + b + c
                if int(tt) < 0:
                    coeffs.append((max(a, 0), max(0, 0), max(0, 0)))
                else:
                    coeffs.append((a, 0, 0))
        except:
            coeffs.append((max(a, 0), max(0, 0), max(0, 0)))
    return coeffs


def fit_quadratic_upper_bounds(f_list):
    """
    Fits a convex quadratic function g_i(t) = a t^2 + b t + c
    such that g_i(t) >= f_i(t) for all integer t in [0, len(f_i) - 1],
    and a >= 0 to ensure convexity.

    Returns: List of (a, b, c) tuples
    """
    coeffs = []

    for f in f_list:
        k = len(f)
        t_vals = np.arange(k)

        # Variables: coefficients of the quadratic function
        a = cp.Variable()
        b = cp.Variable()
        c = cp.Variable()

        # Quadratic function values at integer points
        g_vals = a * t_vals**2 + b * t_vals + c

        # Constraints: g(t) >= f(t) for all t, and a >= 0
        constraints = []

        # Objective: minimize the sum of excess over f
        objective = cp.Minimize(cp.sum((g_vals - f) ** 2))

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)  # Explicitly set a reliable solver

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver failed on function: {f}")

        coeffs.append((a.value, b.value, c.value))

    return coeffs


def plot_twist(f):
    cof = fit_quadratic(f)[0]
    print(cof)
    tt = cof[0] + cof[1] + cof[2]
    print(int(tt))

    # Print polynomial
    print("Approximating quadratic polynomial:")

    # Plot original and approximated function
    x = np.linspace(0, len(f) - 1, 100)
    plt.plot(range(len(f[0])), f[0], "o", label="Original f")
    plt.plot(
        range(len(f[0])),
        [cof[0] * x**2 + cof[1] * x + cof[2] for x in range(len(f[0]))],
        "-",
        label="Quadratic Approximation",
    )
    plt.legend()
    plt.title("Quadratic Approximation of Convex Function")
    plt.grid(True)
    plt.show()
    plt.savefig("plot/plot_twist.png", bbox_inches="tight")


def calc_pwl(functions, demands, all_keys):
    # Check the cnt
    max_parts = 200
    base_log = 2**4
    max_size = 10000000

    max_demand = []
    for key in all_keys:
        max_tmp = 0
        for i in range(len(demands)):
            if key[i] == 0:
                continue
            max_tmp = max(max_tmp, demands[i])
        max_demand.append(max_tmp + 1)

    default_x_list = [0]
    lst_add = 1
    lst_ind = 0
    curr_cnt = base_log
    while lst_ind < max_size:
        for i in range(curr_cnt):
            lst_ind += lst_add
            default_x_list.append(lst_ind)

        if curr_cnt > 1:
            curr_cnt //= 2
        lst_add *= 2

    knots_x_list = []
    knots_y_list = []
    for i in range(len(functions)):
        func = functions[i]
        curr_dems = max_demand[i]
        curr_size = min(len(func), curr_dems + 1)
        curr_func = deepcopy(func[:curr_size])

        pt = 0
        lst = default_x_list[pt]
        curr_x = []
        curr_y = []
        while lst < curr_size - 1:
            curr_x.append(lst)
            curr_y.append(curr_func[lst])

            pt = pt + 1
            lst = default_x_list[pt]

        curr_x.append(curr_size - 1)
        curr_y.append(curr_func[-1])
        knots_x_list.append(curr_x)
        knots_y_list.append(curr_y)

    return knots_x_list, knots_y_list


def run_twist1(dic2, demands):
    all_keys = list(dic2.keys())
    convex_functions = []
    cnt = []
    for key in all_keys:
        cur_q = dic2[key]
        ps = 0
        curr_f = [0]
        for x in cur_q:
            ps += x
            curr_f.append(ps)
        convex_functions.append(curr_f)
        cnt.append(len(curr_f) - 1)
    # plot_twist(convex_functions)

    knots_x_list, knots_y_list = calc_pwl(convex_functions, demands, all_keys)
    n = len(demands)
    m = len(cnt)

    x = [cp.Variable() for i in range(m)]

    # Objective: Minimize sum of piecewise linear approximations of ps[i]
    cost_terms = []
    for i in range(m):
        # Get knots for set i
        knots_x = knots_x_list[i]
        knots_y = knots_y_list[i]
        num_knots = len(knots_x)

        # Ensure knots are valid
        if len(knots_y) != num_knots or num_knots < 2:
            print(i)
            print(all_keys[i])
            print(knots_x)
            print(knots_y)
            raise ValueError(f"Invalid knots for set {i}: need at least 2 knots, got {num_knots}")

        # Compute slopes and intercepts for each segment
        slopes = []
        intercepts = []
        for k in range(num_knots - 1):
            x1, x2 = knots_x[k], knots_x[k + 1]
            y1, y2 = knots_y[k], knots_y[k + 1]
            if x2 <= x1:
                raise ValueError(f"Knots_x for set {i} must be strictly increasing")
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)

        # Piecewise linear function is max of linear functions: slope_k * x + intercept_k
        linear_terms = [slope * x[i] + intercept for slope, intercept in zip(slopes, intercepts)]
        if len(linear_terms) == 1:
            cost_i = linear_terms[0]
        else:
            cost_i = cp.maximum(*linear_terms)
        cost_terms.append(cost_i)

    objective = cp.Minimize(cp.sum(cost_terms))

    # Constraints
    constraints = []
    # Bounds: 0 <= x[i] <= cnt[i]
    for i in range(m):
        constraints.append(x[i] >= 0)
        constraints.append(x[i] <= cnt[i])

    # Demand satisfaction: for each element j, sum of sets covering j meets demand[j]
    for j in range(n):
        coverage = cp.sum([x[i] * all_keys[i][j] for i in range(m)])
        constraints.append(coverage >= demands[j])

    # Formulate and solve the problem
    problem = cp.Problem(objective, constraints)

    # Use a solver for continuous problems (e.g., ECOS, SCS, or default)
    try:
        # problem.solve(solver=cp.ECOS, verbose=True)
        problem.solve()
    except cp.SolverError as e:
        print(f"Solver error: {e}")
        return None, None

    # Check if solution exists
    if problem.status != cp.OPTIMAL:
        print(f"Problem status: {problem.status}")
        return None, None

    # Extract solution
    solution = [x[i].value for i in range(m)]
    optimal_cost = problem.value
    print(optimal_cost)
    return optimal_cost


def run_twist(dic2, demands):
    all_keys = list(dic2.keys())
    convex_functions = []
    cnt = []
    for key in all_keys:
        cur_q = dic2[key]
        ps = 0
        curr_f = [0]
        for x in cur_q:
            ps += x
            curr_f.append(ps)
        convex_functions.append(curr_f)
        cnt.append(len(curr_f) - 1)
    # plot_twist(convex_functions)

    knots_x_list, knots_y_list = calc_pwl(convex_functions, demands, all_keys)
    n = len(demands)
    m = len(cnt)

    prob = pulp.LpProblem("Set_Multicover", pulp.LpMinimize)

    # Decision variables: x[i] is the number of copies of set i (continuous)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=cnt[i], cat="Continuous") for i in range(m)]

    # Auxiliary variables for PWL costs: z[i] represents cost of set i
    z = [pulp.LpVariable(f"z_{i}", lowBound=0, cat="Continuous") for i in range(m)]

    # Objective: Minimize sum of z[i]
    prob += pulp.lpSum(z)

    # Constraints
    # PWL cost constraints
    for i in range(m):
        knots_x = knots_x_list[i]
        knots_y = knots_y_list[i]
        num_knots = len(knots_x)

        # Validate knots
        if len(knots_y) != num_knots or num_knots < 2:
            raise ValueError(f"Invalid knots for set {i}: need at least 2 knots, got {num_knots}")

        # Compute slopes and intercepts
        for k in range(num_knots - 1):
            x1, x2 = knots_x[k], knots_x[k + 1]
            y1, y2 = knots_y[k], knots_y[k + 1]
            if x2 <= x1:
                raise ValueError(f"Knots_x for set {i} must be strictly increasing")
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            # Constraint: z[i] >= slope * x[i] + intercept
            prob += z[i] >= slope * x[i] + intercept, f"cost_constraint_{i}_{k}"

    # Demand satisfaction constraints
    for j in range(n):
        prob += (pulp.lpSum(x[i] * all_keys[i][j] for i in range(m)) >= demands[j], f"demand_{j}")

    # Solve the problem
    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=1))
    except Exception as e:
        print(f"Solver error: {e}")
        return None, None

    # Check if solution exists
    if prob.status != pulp.LpStatusOptimal:
        print(f"Problem status: {pulp.LpStatus[prob.status]}")
        return None, None

    # Extract solution
    solution = [x[i].varValue for i in range(m)]
    optimal_cost = pulp.value(prob.objective)

    optans = 0
    for i in range(len(solution)):
        tmp = int(round(solution[i]))
        optans += convex_functions[i][tmp]

    print(optimal_cost)
    print(optans)
    return optans


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
    solution = x.value
    optan = 0
    for val in solution:
        optan += weights[int(round(val))]
    return prob.value


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
cp_construction_time_avg = []
twist_construction_time_avg = []

res_dp_list_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_cp_list_avg = []
res_twist_list_avg = []

def build_demands(dic, numgroups, n):
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
        l.append(np.random.randint(1, n // 3000 + 2))
    return l
        


demands_org = [np.random.randint(1, 53) for col in all_columns]
range_n = range(10, 21)
n_values = [2**i for i in range_n]
for n in n_values:
    print("n: ", n)
    group_cnt = 21
    columns = all_columns[0:group_cnt]
    data = read_census(columns, n)
    res_dp_list = []
    res_greedy_list = []
    res_lp_list = []
    res_cp_list = []
    res_twist_list = []

    dp_construction_time = []
    lp_construction_time = []
    cp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []
    for i in range(1):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        dic2 = deepcopy(dic)
        demands = build_demands(dic2, group_cnt, n)
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

        # start_time = time.time()
        # res_cp = run_cp(dic2, demands)
        # end_time = time.time()
        # cp_construction_time.append(end_time - start_time)
        # res_cp_list.append(res_cp)

        start_time = time.time()
        res_twist = run_twist(dic2, demands)
        end_time = time.time()
        twist_construction_time.append(end_time - start_time)
        res_twist_list.append(res_twist)

        start_time = time.time()
        res_greedy = run_greedy(dic, demands)
        end_time = time.time()
        res_greedy_list.append(res_greedy)
        greedy_construction_time.append(end_time - start_time)

    lp_construction_time_avg.append(np.mean(lp_construction_time))
    # dp_construction_time_avg.append(np.mean(dp_construction_time))
    greedy_construction_time_avg.append(np.mean(greedy_construction_time))
    # cp_construction_time_avg.append(np.mean(cp_construction_time))
    twist_construction_time_avg.append(np.mean(twist_construction_time))

    res_lp_list_avg.append(np.mean(res_lp_list))
    # res_dp_list_avg.append(np.mean(res_dp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))
    # res_cp_list_avg.append(np.mean(res_cp_list))
    res_twist_list_avg.append(np.mean(res_twist_list))


plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 20})
plt.plot(n_values, lp_construction_time_avg, label="LP", marker="o")
# plt.plot(n_values, dp_construction_time_avg, label="DP", marker="s")
plt.plot(n_values, greedy_construction_time_avg, label="Greedy", marker="^")
# plt.plot(n_values, cp_construction_time_avg, label="CP", marker="X")
plt.plot(n_values, twist_construction_time_avg, label="Twist", marker="x")
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
plt.bar(x - width, res_twist_list_avg, width, label="Twist", color="red")
plt.bar(x, res_greedy_list_avg, width, label="Greedy", color="orange")
plt.bar(x + width, res_lp_list_avg, width, label="LP", color="blue")
# plt.bar(x + 2 * width, res_cp_list_avg, width, label="CP", color="green")

plt.xticks(x, labels=[r"$2^{{{}}}$".format(i) for i in range_n])
plt.xlabel("Number of Records")
plt.ylabel("Total Weight")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.savefig("plot/comparison_varying_n.png", bbox_inches="tight")


lp_construction_time_avg = []
dp_construction_time_avg = []
greedy_construction_time_avg = []
cp_construction_time_avg = []
twist_construction_time_avg = []
res_dp_list_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_cp_list_avg = []
res_twist_list_avg = []

demands_org = [np.random.randint(1, 53) for col in all_columns]
num_groups = range(3, 68, 5)
n_size = 100000
for num_group in num_groups:
    columns = all_columns[0:num_group]
    print("n_groups: ", num_group)
    data = read_census(columns, n_size)
    res_dp_list = []
    res_greedy_list = []
    res_lp_list = []
    res_cp_list = []
    res_twist_list = []
    dp_construction_time = []
    lp_construction_time = []
    cp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []

    for i in range(1):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        dic2 = deepcopy(dic)
        print("PQ size:", len(dic))
        demands = build_demands(dic2, num_group, n_size)
        start_time = time.time()
        res_lp = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)

        # start_time = time.time()
        # res_cp = run_cp(dic2, demands)
        # end_time = time.time()
        # cp_construction_time.append(end_time - start_time)
        # res_cp_list.append(res_cp)

        start_time = time.time()
        res_twist = run_twist(dic2, demands)
        # res_twist = 0
        end_time = time.time()
        twist_construction_time.append(end_time - start_time)
        res_twist_list.append(res_twist)

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
    # dp_construction_time_avg.append(np.mean(dp_construction_time))
    greedy_construction_time_avg.append(np.mean(greedy_construction_time))
    cp_construction_time_avg.append(np.mean(cp_construction_time))
    twist_construction_time_avg.append(np.mean(twist_construction_time))

    res_lp_list_avg.append(np.mean(res_lp_list))
    # res_dp_list_avg.append(np.mean(res_dp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))
    res_cp_list_avg.append(np.mean(res_cp_list))
    res_twist_list_avg.append(np.mean(res_twist_list))


plt.figure(figsize=(10, 6))
plt.rcParams.update({"font.size": 20})
plt.plot(num_groups, lp_construction_time_avg, label="LP", marker="o")
# plt.plot(n_values, dp_construction_time_avg, label="DP", marker="s")
plt.plot(num_groups, greedy_construction_time_avg, label="Greedy", marker="^")
# plt.plot(num_groups, cp_construction_time_avg, label="CP", marker="X")
plt.plot(num_groups, twist_construction_time_avg, label="Twist", marker="x")


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
plt.bar(x - width, res_twist_list_avg, width, label="Twist", color="red")
plt.bar(x, res_greedy_list_avg, width, label="Greedy", color="orange")
plt.bar(x + width, res_lp_list_avg, width, label="LP", color="blue")
# plt.bar(x + 2 * width, res_cp_list_avg, width, label="CP", color="green")

plt.xticks(x, num_groups)
plt.xlabel("Number of Groups")
plt.ylabel("Total Weight")
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.savefig("plot/comparison_varying_group.png", bbox_inches="tight")

# Prompt:
# I have the following problem that is modeled as a convex optimization problem. Write the code to solve this problem using python's cvxpy package. I want to solve an instance of the weighted set multicover problem. I have a demand requirement for each of the numbers from 1 to n. demands[i] stores how many times I need to cover the number i. I have a collection of sets. For each set j, the elements it covers are stored in a boolean tuple of length n, which indicates if each element is included in this set or not. all_keys[j] stores the boolean tuple representing the set j. From each set j there are cnt[j] available copies that can be used. The cost of using the j'th set k times is stored in convex_functions[j][k] for all 0 <= k <= cnt[j]. I want to find the minimum cost to satisfy all the demands.
# I want to minimize the following objective function:
# \sum(f_i(x_i))
# I have an array of piecewise linear functions stored in a list called convex_functions. Each element of the convex_function is itself a list.
