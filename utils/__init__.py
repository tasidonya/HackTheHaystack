import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


def get_unique_values(file_paths, col_name, **kwargs):
    """
    :param file_paths: list: list of file paths to look in
    :param col_name: the column name to get unique values for
    :param kwargs: any kwargs for pandas.DataFrame() e.g. sep=","
    :return: numpy.ndarray
    """
    df = pd.DataFrame(columns=[col_name])
    for path in file_paths:
        tmp = pd.read_csv(path, **kwargs)
        df = df.append(tmp, ignore_index=True, sort=False)
    return df[col_name].unique()


def get_email(file_path):
    """
    :param file_path:
    :return:
    """
    pattern = '%m/%d/%Y %H:%M:%S'
    columns = ["epoch", "freq"]
    df = pd.read_csv(file_path)
    df = df.where((pd.notnull(df)), None)
    df = df[df.activity == "Send"]                      # Only keep emails which are sent (remove view and receive)
    users = df.user.unique()                            # Get a list of unique user names
    # Set up data dict
    data = {user: pd.DataFrame(columns=columns) for user in users}
    # Get frequency of each email address
    freq = {}
    for _, row in df.iterrows():
        for address in row.to.split(";"):
            if address not in freq:
                freq.update({address: 1})
            else:
                freq[address] += 1
    n = sum([value for value in freq.values()])
    for _, row in df.iterrows():
        f = np.asarray([freq[address] for address in row.to.split(";")])
        f = np.max(- np.log(f / n))
        epoch = int(time.mktime(time.strptime(row.date, pattern)))
        tmp = pd.DataFrame([[epoch, f]], columns=columns)
        data[row.user] = data[row.user].append(tmp)
    return data


if __name__ == "__main__":
    data = get_email("../data/haystack_email.csv")
    for user, d in data.items():
        print(user)
        print(d)
        print("-----------")



