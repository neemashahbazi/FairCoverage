from algorithms import run_greedy, run_lp, run_twist, construct_priority_queues, vizualize, visualize_rss, rss_demands,run_lp_randomized_rounding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
plt.rcParams["font.family"] = "serif"


def read_music(columns, n, n_groups):
    data = pd.read_csv("data/amazon_with_genres.csv").head(n)
    weight = data["weight"]
    data = data[columns[1 : n_groups + 1]]
    data["weight"] = weight.astype(int)
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
    "Dance & Electronic",
    "Pop",
    "Country",
    "Rock",
    "Rap & Hip-Hop",
    "Alternative Rock",
    "Electronica",
    "Soundtracks",
    "International",
    "House",
    "R&B",
    "Miscellaneous",
    "Latin Music",
    "Gangsta & Hardcore",
    "Classical",
    "Children 's Music",
    "Jazz",
    "Caribbean & Cuba",
    "Reggae",
    "Folk",
    "Christian",
    "Dubstep",
    "Indie & Lo-Fi",
    "Hard Rock & Metal",
    "Trance",
    "Latin Hip-Hop",
    "Holiday",
    "Southern Rap",
    "Soul",
    "Country Rock",
    "Classic Rock",
    "Blues",
    "Christmas",
    "Techno",
    "Pop Rock",
    "Singer-Songwriters",
    "Christian Contemporary Music",
    "Hardcore & Punk",
    "Traditional Country",
    "Gospel",
    "Contemporary R&B",
    "West Coast",
    "Bluegrass",
    "Pop Rap",
    "Easy Listening",
    "Hard Rock",
    "Drum & Bass",
    "East Coast",
    "Contemporary Country",
    "Soft Rock",
    "Children 's",
    "Folk Rock",
    "Adult Alternative",
    "New Wave & Post-Punk",
    "Vocal Jazz",
    "New Age",
    "Traditional Folk",
    "Thrash & Speed Metal",
    "Alt-Country & Americana",
    "Dance Pop",
    "Swing Jazz",
    "Adult Contemporary",
    "Comedy",
    "Far East & Asia",
    "Traditional Jazz & Ragtime",
    "Jazz Fusion",
    "Alternative Metal",
    "Smooth Jazz",
    "Progressive",
    "Modern Blues",
    "Rap Rock",
    "Opera & Vocal",
    "Europe",
    "Trip-Hop",
    "Latin Pop",
    "International Rap",
    "Cowboy",
    "Blues Rock",
    "Traditional British & Celtic Folk",
    "American Alternative",
    "Contemporary Folk",
    "Broadway & Vocalists",
    "Country & Bluegrass",
    "Classic R&B",
    "Electric Blues Guitar",
    "India & Pakistan",
    "Oldies",
    "Hanukkah",
    "Experimental Rap",
    "Bebop",
    "Contemporary Blues",
    "Modern Postbebop",
    "Celtic",
    "Traditional Vocal Pop",
    "Vocal Pop",
    "Funk",
    "Avant Garde & Free Jazz",
    "Roots Rock",
    "Big Band",
    "North America",
    "Exercise",
    "Latin Rock",
    "Disco",
    "Outlaw & Progressive Country",
    "Alternative",
    "Death Metal",
    "Southern Rock",
    "Praise & Worship",
    "Middle East",
    "Oldies & Retro",
    "Honky-Tonk",
    "New Orleans Jazz",
    "Goth & Industrial",
    "Spoken Word & Interviews",
    "Poetry",
    "Bass",
    "Banda",
    "Latin Christian",
    "Acoustic Blues",
    "Lullabies",
    "Cabaret",
    "Country Gospel",
    "Australia & New Zealand",
    "Old School",
    "Educational",
    "Norteno",
    "Sound Effects",
    "Euro Pop",
    "Environmental",
    "Turntablists",
    "Jam Bands",
    "Neo-Soul",
    "Reggaeton",
    "Meditation",
    "6:14",
    "Mambo",
    "6:05",
    "Soul-Jazz & Boogaloo",
    "Salsa",
]

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
lp_randomized_construction_time_avg = []

res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []
res_lp_randomized_list_avg = []


range_n = range(10, 15)
n_values = [2**i for i in range_n]
for n in n_values:
    print("n: ", n)
    group_cnt = 20
    data = read_music(all_columns, n, group_cnt)
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
        demands = build_demands(dic2, group_cnt)
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
    "music",
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

demands_org = [np.random.randint(1, 10) for col in all_columns]
num_groups = range(3, 140, 10)
n_size = 50000
for num_group in num_groups:
    print("n_groups: ", num_group)
    data = read_music(all_columns, n_size, num_group)
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
        demands = build_demands(dic2, num_group)
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
    "music",
    "group",
)

visualize_rss(num_groups, lp_rss_avg, greedy_rss_avg, twist_rss_avg, lp_randomized_rss_avg, "music")

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
lp_randomized_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []
res_lp_randomized_list_avg = []

num_group = 20
n_size = 50000
distributions = ["random", "uniform", "normal", "exponential", "poisson", "zipf"]
for dist in distributions:
    print("distributions: ", dist)
    data = read_music(all_columns, n_size, num_group)
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
        demands = build_demands(dic2, num_group, dist)
        start_time = time.time()
        res_lp, _ = run_lp(data, demands)
        end_time = time.time()
        lp_construction_time.append(end_time - start_time)
        res_lp_list.append(res_lp)

        start_time = time.time()
        res_lp_randomized, covered_demands_lp_randomized = run_lp_randomized_rounding(data, demands)
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
    "music",
    "distribution",
)
