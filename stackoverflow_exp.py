from algorithms import run_greedy, run_lp, run_twist, construct_priority_queues, vizualize, visualize_rss, rss_demands, run_lp_randomized_rounding
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
    "LearnCode_Other online resources (e.g., videos, blogs, forum, online community)",
    "Employment_Employed, full-time",
    "CodingActivities_Hobby",
    "TechDoc_API document(s) and/or SDK document(s)",
    "BuyNewTool_Start a free trial",
    "BuyNewTool_Ask developers I know/work with",
    "TechDoc_User guides or README files found in the source repository",
    "LearnCode_Books / Physical media",
    "LearnCode_Online Courses or Certification",
    "LearnCode_School (i.e., University, College, etc)",
    "TechDoc_Traditional public search engine",
    "BuyNewTool_Visit developer communities like Stack Overflow",
    "LearnCode_On the job training",
    "BuildvsBuy_Is ready-to-go but also customizable for growth and targeted use cases",
    "EdLevel_Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
    "Age_25-34 years old",
    "RemoteWork_Hybrid (some remote, some in-person)",
    "CodingActivities_Professional development or self-paced learning from online courses",
    "RemoteWork_Remote",
    "PurchaseInfluence_I have some influence",
    "DevType_Developer, full-stack",
    "PurchaseInfluence_I have little or no influence",
    "EdLevel_Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
    "TechDoc_First-party knowledge base",
    "Age_35-44 years old",
    "LearnCode_Colleague",
    "BuyNewTool_Read ratings or reviews on third party sites like G2 Crowd",
    "Age_18-24 years old",
    "CodingActivities_Contribute to open-source projects",
    "TechDoc_AI-powered search/dev tool (free)",
    "RemoteWork_In-person",
    "Employment_Independent contractor, freelancer, or self-employed",
    "CodingActivities_Freelance/contract work",
    "PurchaseInfluence_I have a great deal of influence",
    "DevType_Developer, back-end",
    "OrgSize_20 to 99 employees",
    "BuildvsBuy_Is set up to be customized and needs to be engineered into a usable product",
    "OrgSize_100 to 499 employees",
    "Employment_Student, full-time",
    "BuildvsBuy_Out-of-the-box is ready to go with little need for customization",
    "BuyNewTool_Ask a generative AI tool",
    "CodingActivities_Bootstrapping a business",
    "EdLevel_Some college/university study without earning a degree",
    "CodingActivities_School or academic work",
    "MainBranch_I am not primarily a developer, but I write code sometimes as part of my work/studies",
    "CodingActivities_I don’t code outside of work",
    "LearnCode_Coding Bootcamp",
    "LearnCode_Friend or family member",
    "Age_45-54 years old",
    "BuyNewTool_Research companies that have advertised on sites I visit",
    "EdLevel_Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
    "OrgSize_10,000 or more employees",
    "TechDoc_AI-powered search/dev tool (paid)",
    "OrgSize_1,000 to 4,999 employees",
    "DevType_Student",
    "OrgSize_2 to 9 employees",
    "YearsCode_10",
    "YearsCodePro_2",
    "Employment_Employed, part-time",
    "YearsCodePro_3",
    "OrgSize_10 to 19 employees",
    "Employment_Not employed, but looking for work",
    "MainBranch_I am learning to code",
    "YearsCode_5",
    "LearnCode_Other (please specify):",
    "YearsCodePro_5",
    "YearsCode_6",
    "YearsCode_8",
    "DevType_Developer, front-end",
    "MainBranch_I code primarily as a hobby",
    "YearsCode_7",
    "YearsCode_4",
    "YearsCodePro_10",
    "YearsCodePro_4",
    "OrgSize_500 to 999 employees",
    "BuyNewTool_Other (please specify):",
    "OrgSize_Just me - I am a freelancer, sole proprietor, etc.",
    "EdLevel_Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
    "YearsCodePro_Less than 1 year",
    "YearsCodePro_6",
    "YearsCode_15",
    "Employment_Student, part-time",
    "YearsCodePro_1",
    "YearsCode_20",
    "Age_55-64 years old",
    "Age_Under 18 years old",
    "YearsCode_12",
    "YearsCodePro_8",
    "YearsCode_3",
    "YearsCodePro_7",
    "DevType_Developer, desktop or enterprise applications",
    "DevType_Other (please specify):",
    "YearsCode_9",
    "BuyNewTool_Research companies that have emailed me",
    "DevType_Developer, mobile",
    "OrgSize_5,000 to 9,999 employees",
    "YearsCode_14",
    "EdLevel_Associate degree (A.A., A.S., etc.)",
    "YearsCodePro_12",
    "YearsCode_25",
    "YearsCode_2",
    "YearsCode_11",
    "YearsCodePro_15",
    "DevType_Developer, embedded applications or devices",
    "YearsCodePro_20",
    "MainBranch_I used to be a developer by profession, but no longer am",
    "YearsCodePro_9",
    "YearsCode_13",
    "YearsCode_30",
    "YearsCode_16",
    "YearsCodePro_11",
    "DevType_Engineering manager",
    "DevType_Academic researcher",
    "YearsCode_18",
    "Employment_Not employed, and not looking for work",
    "EdLevel_Primary/elementary school",
    "YearsCodePro_13",
    "DevType_Data engineer",
    "CodingActivities_Other (please specify):",
    "YearsCodePro_14",
    "OrgSize_I don’t know",
    "YearsCode_17",
    "DevType_Data scientist or machine learning specialist",
    "DevType_DevOps specialist",
    "YearsCodePro_25",
    "YearsCode_40",
    "YearsCodePro_16",
    "DevType_Research & Development role",
    "EdLevel_Something else",
    "YearsCode_24",
    "YearsCodePro_18",
    "YearsCode_22",
    "DevType_Senior Executive (C-Suite, VP, etc.)",
    "YearsCodePro_17",
    "Age_65 years or older",
    "YearsCode_35",
    "YearsCode_1",
    "DevType_Developer, game or graphics",
    "YearsCodePro_30",
    "Employment_Retired",
    "DevType_Cloud infrastructure engineer",
    "YearsCode_23",
    "YearsCodePro_24",
    "YearsCode_26",
    "TechDoc_Other (please specify):",
    "YearsCode_Less than 1 year",
    "YearsCode_19",
    "DevType_System administrator",
    "Employment_I prefer not to say",
    "DevType_Developer, AI",
    "DevType_Developer, QA or test",
    "DevType_Data or business analyst",
    "YearsCode_21",
    "YearsCodePro_19",
    "YearsCode_28",
    "YearsCode_27",
    "YearsCodePro_22",
    "YearsCodePro_23",
    "YearsCodePro_26",
    "DevType_Project manager",
    "YearsCodePro_27",
    "YearsCodePro_21",
    "DevType_Security professional",
    "DevType_Educator",
    "YearsCodePro_28",
    "DevType_Scientist",
    "YearsCode_32",
    "Age_Prefer not to say",
    "DevType_Engineer, site reliability",
    "YearsCode_34",
    "DevType_Product manager",
    "YearsCode_42",
    "YearsCode_38",
    "YearsCodePro_35",
    "YearsCode_29",
    "YearsCode_45",
    "YearsCode_More than 50 years",
    "DevType_Blockchain",
    "DevType_Developer Experience",
    "YearsCode_36",
    "YearsCode_33",
    "YearsCode_37",
    "DevType_Hardware Engineer",
    "YearsCodePro_29",
    "YearsCode_44",
    "YearsCodePro_32",
    "YearsCodePro_40",
    "DevType_Designer",
    "YearsCode_43",
    "DevType_Database administrator",
    "YearsCodePro_34",
    "YearsCode_41",
    "YearsCode_31",
    "YearsCode_39",
    "YearsCodePro_38",
    "YearsCodePro_33",
    "YearsCodePro_36",
    "YearsCodePro_31",
    "DevType_Developer Advocate",
    "YearsCodePro_37",
    "YearsCode_46",
    "DevType_Marketing or sales professional",
    "YearsCode_50",
    "YearsCode_48",
    "YearsCode_47",
    "YearsCodePro_45",
    "YearsCodePro_42",
    "YearsCodePro_39",
    "YearsCodePro_41",
    "YearsCodePro_More than 50 years",
    "YearsCodePro_44",
    "YearsCodePro_43",
    "YearsCode_49",
    "YearsCodePro_46",
    "YearsCodePro_48",
    "YearsCodePro_50",
    "YearsCodePro_49",
    "YearsCodePro_47",
]

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
lp_randomized_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []
res_lp_randomized_list_avg = []


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
    "stackoverflow",
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

num_groups = range(3, 49, 10)
n_size = 50000
for num_group in num_groups:
    columns = all_columns[0:num_group]
    print("n_groups: ", num_group)
    data = read_stackoverflow(columns, n_size)
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
    "stackoverflow",
    "group",
)

visualize_rss(num_groups, lp_rss_avg, greedy_rss_avg, twist_rss_avg, lp_randomized_rss_avg, "stackoverflow")

lp_construction_time_avg = []
greedy_construction_time_avg = []
twist_construction_time_avg = []
lp_randomized_construction_time_avg = []
res_greedy_list_avg = []
res_lp_list_avg = []
res_twist_list_avg = []
res_lp_randomized_list_avg = []

num_group = 40
n_size = 50000
distributions = ["random", "uniform", "normal", "exponential", "poisson", "zipf"]
for dist in distributions:
    columns = all_columns[:num_group]
    print("distribution: ", dist)
    data = read_stackoverflow(columns, n_size)
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
    "stackoverflow",
    "distribution",
)
