from algorithms import run_greedy, run_lp, run_twist, construct_priority_queues, vizualize, visualize_rss, rss_demands
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

plt.rcParams["font.family"] = "serif"


def build_weights(data):
    weights = []
    for _, row in data.iterrows():
        count_ones = sum([1 for val in row if val == 1])
        rnd = np.random.randint(1, 10)
        epsilon = 1.0
        if rnd < 3:
            count_ones += epsilon
        weights.append(count_ones)
    return weights


def read_stackoverflow(columns, n):
    data = pd.read_csv("data/preprocessed_survey_results.csv").head(n)
    data = data[columns]
    data["weight"] = build_weights(data)
    # data["weight"] = np.random.randint(1, 1000, size=len(data))
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
            tmp = 5
        elif distribution == "normal":
            tmp = int(np.random.normal(5, 2))
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
    "MainBranch_I am a developer by profession",
    "MainBranch_I code primarily as a hobby",
    "MainBranch_I am learning to code",
    "MainBranch_I used to be a developer by profession, but no longer am",
    "MainBranch_I am not primarily a developer, but I write code sometimes as part of my work/studies",
    "Age_25-34 years old",
    "Age_Prefer not to say",
    "Age_45-54 years old",
    "Age_55-64 years old",
    "Age_35-44 years old",
    "Age_65 years or older",
    "Age_18-24 years old",
    "Age_Under 18 years old",
    "Employment_Independent contractor, freelancer, or self-employed",
    "Employment_Retired",
    "Employment_Not employed, and not looking for work",
    "Employment_Employed, full-time",
    "Employment_Employed, part-time",
    "Employment_Student, full-time",
    "Employment_Not employed, but looking for work",
    "Employment_I prefer not to say",
    "Employment_Student, part-time",
    "RemoteWork_Remote",
    "RemoteWork_Hybrid (some remote, some in-person)",
    "RemoteWork_In-person",
    "CodingActivities_Contribute to open-source projects",
    "CodingActivities_I don’t code outside of work",
    "CodingActivities_Other (please specify):",
    "CodingActivities_Hobby",
    "CodingActivities_Freelance/contract work",
    "CodingActivities_Bootstrapping a business",
    "CodingActivities_Professional development or self-paced learning from online courses",
    "CodingActivities_School or academic work",
    "EdLevel_Something else",
    "EdLevel_Primary/elementary school",
    "EdLevel_Associate degree (A.A., A.S., etc.)",
    "EdLevel_Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
    "EdLevel_Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
    "EdLevel_Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
    "EdLevel_Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
    "EdLevel_Some college/university study without earning a degree",
    "LearnCode_On the job training",
    "LearnCode_Friend or family member",
    "LearnCode_Coding Bootcamp",
    "LearnCode_School (i.e., University, College, etc)",
    "LearnCode_Online Courses or Certification",
    "LearnCode_Other (please specify):",
    "LearnCode_Books / Physical media",
    "LearnCode_Colleague",
    "LearnCode_Other online resources (e.g., videos, blogs, forum, online community)",
    "TechDoc_AI-powered search/dev tool (paid)",
    "TechDoc_User guides or README files found in the source repository",
    "TechDoc_Other (please specify):",
    "TechDoc_API document(s) and/or SDK document(s)",
    "TechDoc_AI-powered search/dev tool (free)",
    "TechDoc_Traditional public search engine",
    "TechDoc_First-party knowledge base",
    "DevType_Product manager",
    "DevType_Cloud infrastructure engineer",
    "DevType_Engineering manager",
    "DevType_Designer",
    "DevType_Developer, QA or test",
    "DevType_Student",
    "DevType_Developer, AI",
    "DevType_Engineer, site reliability",
    "DevType_Data scientist or machine learning specialist",
    "DevType_Developer Experience",
    "DevType_Academic researcher",
    "DevType_Senior Executive (C-Suite, VP, etc.)",
    "DevType_Educator",
    "DevType_Marketing or sales professional",
    "DevType_Developer, embedded applications or devices",
    "DevType_Security professional",
    "DevType_Research & Development role",
    "DevType_Data engineer",
    "DevType_Developer, desktop or enterprise applications",
    "DevType_DevOps specialist",
    "DevType_Developer, front-end",
    "DevType_Blockchain",
    "DevType_Developer, full-stack",
    "DevType_System administrator",
    "DevType_Developer Advocate",
    "DevType_Developer, game or graphics",
    "DevType_Data or business analyst",
    "DevType_Project manager",
    "DevType_Scientist",
    "DevType_Other (please specify):",
    "DevType_Database administrator",
    "DevType_Hardware Engineer",
    "DevType_Developer, back-end",
    "DevType_Developer, mobile",
    "OrgSize_1,000 to 4,999 employees",
    "OrgSize_500 to 999 employees",
    "OrgSize_I don’t know",
    "OrgSize_20 to 99 employees",
    "OrgSize_10 to 19 employees",
    "OrgSize_10,000 or more employees",
    "OrgSize_100 to 499 employees",
    "OrgSize_Just me - I am a freelancer, sole proprietor, etc.",
    "OrgSize_2 to 9 employees",
    "OrgSize_5,000 to 9,999 employees",
    "PurchaseInfluence_I have some influence",
    "PurchaseInfluence_I have little or no influence",
    "PurchaseInfluence_I have a great deal of influence",
    "BuyNewTool_Visit developer communities like Stack Overflow",
    "BuyNewTool_Research companies that have emailed me",
    "BuyNewTool_Other (please specify):",
    "BuyNewTool_Read ratings or reviews on third party sites like G2 Crowd",
    "BuyNewTool_Start a free trial",
    "BuyNewTool_Ask a generative AI tool",
    "BuyNewTool_Ask developers I know/work with",
    "BuyNewTool_Research companies that have advertised on sites I visit",
    "BuildvsBuy_Is ready-to-go but also customizable for growth and targeted use cases",
    "BuildvsBuy_Is set up to be customized and needs to be engineered into a usable product",
    "BuildvsBuy_Out-of-the-box is ready to go with little need for customization",
]

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []

range_n = range(10, 16)
n_values = [2**i for i in range_n]
for n in n_values:
    print("n: ", n)
    group_cnt = 40
    columns = all_columns[0:group_cnt]
    data = read_stackoverflow(columns, n)
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
    "stackoverflow",
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

num_groups = range(3, 49, 10)
n_size = 50000
for num_group in num_groups:
    columns = all_columns[0:num_group]
    print("n_groups: ", num_group)
    data = read_stackoverflow(columns, n_size)
    res_greedy_list = []
    res_lp_list = []
    res_twist_list = []
    lp_construction_time = []
    greedy_construction_time = []
    twist_construction_time = []
    lp_rss = []
    twist_rss = []
    greedy_rss = []

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
    "stackoverflow",
    "group",
)

visualize_rss(
    num_groups,
    lp_rss_avg,
    greedy_rss_avg,
    twist_rss_avg,
    "stackoverflow"
)

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []

num_group = 40
n_size = 50000
distributions = ["random", "uniform", "normal", "exponential", "poisson", "zipf"]
for dist in distributions:
    columns = all_columns[:num_group]
    print("n_groups: ", num_group)
    data = read_stackoverflow(columns, n_size)
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
    "stackoverflow",
    "distribution",
)
