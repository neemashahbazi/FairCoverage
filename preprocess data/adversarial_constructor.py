import pandas as pd

def create_custom_dataset(d):
    data = []
    for i in range(1, d + 2):
        row = [1 if j < i else 0 for j in range(d)]
        if i <= d:
            weight = 1 / (d - i + 1)
        else:
            weight = 1.1
        #weight = int(weight * 100)
        row.append(weight)
        data.append(row)
    columns = [f'attr_{j+1}' for j in range(d)] + ['weight']
    df = pd.DataFrame(data, columns=columns)
    return df

# Example usage:
d = 6400
df = create_custom_dataset(d)
df.to_csv("data/synthetic.csv", index=False)