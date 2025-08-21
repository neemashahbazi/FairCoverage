from algorithms import run_greedy, run_lp, run_twist, construct_priority_queues, vizualize, visualize_rss, rss_demands
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
plt.rcParams["font.family"] = "serif"


def read_yelp(columns, n):
    data = pd.read_csv("data/yelp_business_sorted.csv").head(n)
    data = data[columns + ["review_count"]]
    data["weight"] = data["review_count"].astype(int)
    data = data.drop(columns=["review_count"])
    return data


def build_demands(dic, numgroups, distribution="random"):
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
            tmp = np.random.randint(1, 10)
        elif distribution == "uniform":
            tmp = 10
        elif distribution == "normal":
            tmp = int(np.random.normal(5, 1))
        elif distribution == "exponential":
            tmp = int(np.random.exponential(5))
        elif distribution == "poisson":
            tmp = int(np.random.poisson(5))
        elif distribution == "zipf":
            tmp = int(np.random.zipf(2))
        mn = cnt_sets[i]
        l.append(min(tmp, mn))
    return l


all_columns = [
    "Restaurants",
    "Food",
    "Shopping",
    "Home Services",
    "Beauty & Spas",
    "Nightlife",
    "Health & Medical",
    "Local Services",
    "Bars",
    "Automotive",
    "Event Planning & Services",
    "Sandwiches",
    "American (Traditional)",
    "Active Life",
    "Pizza",
    "Coffee & Tea",
    "Fast Food",
    "Breakfast & Brunch",
    "American (New)",
    "Hotels & Travel",
    "Home & Garden",
    "Fashion",
    "Burgers",
    "Arts & Entertainment",
    "Auto Repair",
    "Hair Salons",
    "Nail Salons",
    "Mexican",
    "Italian",
    "Specialty Food",
    "Doctors",
    "Pets",
    "Real Estate",
    "Seafood",
    "Fitness & Instruction",
    "Professional Services",
    "Hair Removal",
    "Desserts",
    "Chinese",
    "Bakeries",
    "Grocery",
    "Salad",
    "Hotels",
    "Chicken Wings",
    "Cafes",
    "Ice Cream & Frozen Yogurt",
    "Caterers",
    "Pet Services",
    "Dentists",
    "Skin Care",
    "Venues & Event Spaces",
    "Tires",
    "Wine & Spirits",
    "Beer",
    "Delis",
    "Oil Change Stations",
    "Waxing",
    "Contractors",
    "Women's Clothing",
    "Massage",
    "Sports Bars",
    "Day Spas",
    "General Dentistry",
    "Education",
    "Flowers & Gifts",
    "Auto Parts & Supplies",
    "Apartments",
    "Convenience Stores",
    "Home Decor",
    "Gyms",
    "Japanese",
    "Pubs",
    "Cocktail Bars",
    "Sushi Bars",
    "Barbeque",
    "Juice Bars & Smoothies",
    "Barbers",
    "Car Dealers",
    "Sporting Goods",
    "Accessories",
    "Cosmetic Dentists",
    "Drugstores",
    "Local Flavor",
    "Furniture Stores",
    "Pet Groomers",
    "Asian Fusion",
    "Cosmetics & Beauty Supply",
    "Jewelry",
    "Steakhouses",
    "Diners",
]

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []

res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []


range_n = range(10, 18)
n_values = [2**i for i in range_n]
for n in n_values:
    print("n: ", n)
    group_cnt = 20
    columns = all_columns[0:group_cnt]
    data = read_yelp(columns, n)
    res_greedy_list = []
    res_lp_list = []
    res_twist_list = []

    lp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []
    for i in range(1):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        dic2 = deepcopy(dic)
        demands = build_demands(dic2, group_cnt)
        start_time = time.time()
        res_lp, _ = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)

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

    res_lp_list_avg.append(np.mean(res_lp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))
    res_twist_list_avg.append(np.mean(res_twist_list))


vizualize(
    n_values,
    res_lp_list_avg,
    res_greedy_list_avg,
    res_twist_list_avg,
    lp_construction_time_avg,
    greedy_construction_time_avg,
    twist_construction_time_avg,
    "Number of Records",
    "Total Weight",
    "Time (sec)",
    "yelp",
    "n",
)

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []
lp_rss_avg = []
twist_rss_avg = []
greedy_rss_avg = []

demands_org = [np.random.randint(1, 10) for col in all_columns]
num_groups = range(3, 80, 10)
n_size = 100000
for num_group in num_groups:
    columns = all_columns[0:num_group]
    print("n_groups: ", num_group)
    data = read_yelp(columns, n_size)
    res_greedy_list = []
    res_lp_list = []
    res_twist_list = []
    lp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []
    lp_rss = []
    twist_rss = []
    greedy_rss = []

    for i in range(1):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        dic2 = deepcopy(dic)
        print("PQ size:", len(dic))
        demands = build_demands(dic2, num_group)
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
    lp_rss_avg.append(np.mean(lp_rss))
    twist_rss_avg.append(np.mean(twist_rss))
    greedy_rss_avg.append(np.mean(greedy_rss))

    res_lp_list_avg.append(np.mean(res_lp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))
    res_twist_list_avg.append(np.mean(res_twist_list))


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
    "yelp",
    "group",
)

visualize_rss(
    num_groups,
    lp_rss_avg,
    greedy_rss_avg,
    twist_rss_avg,
    "yelp",
)

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []

num_group = 20
n_size = 100000
distributions = ["random", "uniform", "normal", "exponential", "poisson", "zipf"]
for dist in distributions:
    columns = all_columns[:num_group]
    print("distribution: ", dist)
    data = read_yelp(columns, n_size)
    res_greedy_list = []
    res_lp_list = []
    res_twist_list = []
    lp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []
    for i in range(5):
        print("run:", i + 1)
        dic = construct_priority_queues(data)
        dic2 = deepcopy(dic)
        print("PQ size:", len(dic))
        demands = build_demands(dic2, num_group, dist)
        start_time = time.time()
        res_lp, _ = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)

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

    res_lp_list_avg.append(np.mean(res_lp_list))
    res_greedy_list_avg.append(np.mean(res_greedy_list))
    res_twist_list_avg.append(np.mean(res_twist_list))

vizualize(
    distributions,
    res_lp_list_avg,
    res_greedy_list_avg,
    res_twist_list_avg,
    lp_construction_time_avg,
    greedy_construction_time_avg,
    twist_construction_time_avg,
    "Distribution of Demands",
    "Total Weight",
    "Time (sec)",
    "yelp",
    "distribution",
)
