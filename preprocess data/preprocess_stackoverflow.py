import pandas as pd

columns = [
    "MainBranch",
    "Age",
    "Employment",
    "RemoteWork",
    "CodingActivities",
    "EdLevel",
    "LearnCode",
    "TechDoc",
    "YearsCode",
    "YearsCodePro",
    "DevType",
    "OrgSize",
    "PurchaseInfluence",
    "BuyNewTool",
    "BuildvsBuy",
]

df = pd.read_csv("data/survey_results_public.csv")[columns]

for col in df.columns:
    values = df[col].dropna().astype(str).str.split(";")
    unique_values = set([item.strip() for sublist in values for item in sublist])
    for val in unique_values:
        if val == "":
            continue
        df[f"{col}_{val}"] = df[col].astype(str).apply(lambda x: int(val in [i.strip() for i in x.split(";")]))        

df = df.drop(columns, axis=1)
df.to_csv("data/preprocessed_survey_results.csv", index=False)