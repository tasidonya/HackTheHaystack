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


def sort_data(data):
    """
    :param data:
    :return:
    """
    for user, item in data.items():
        data[user] = item.sort_values(by=["epoch"])
    return data


def normalise_data(data, columns):
    """
    :param data:
    :return:
    """
    maximum = [0 for _ in columns]
    for user, df in data.items():
        for i, (col, m) in enumerate(zip(columns, maximum)):
            col_max = df[col].max()
            if col_max > m:
                maximum[i] = col_max
    for user, df in data.items():
        for col, m in zip(columns, maximum):
            df[col] /= m
    return data


def get_email_freq(df):
    """
    :param df:
    :return:
    """
    freq1 = {}
    freq2 = {}
    for _, row in df.iterrows():
        for address in row.to.split(";"):
            domain = address.split("@")[1]
            if address not in freq1:
                freq1.update({address: 1})
            else:
                freq1[address] += 1
            if domain not in freq2:
                freq2.update({domain: 1})
            else:
                freq2[domain] += 1
    return freq1, freq2


def get_f1(freq, row, n):
    """
    :param freq:
    :param row:
    :param n:
    :return:
    """
    f = np.asarray([freq[address] for address in row.to.split(";")])
    return np.max(- np.log(f / n))


def get_f2(freq, row, n):
    """
    :param freq:
    :param row:
    :param n:
    :return:
    """
    f = np.asarray([freq[address.split("@")[1]] for address in row.to.split(";")])
    return np.max(- np.log(f / n))


def get_f(f_values):
    """
    :param f_values:
    :return:
    """
    return (np.prod(f_values) * np.sum(f_values)) / len(f_values)


def calculate_combined_feature(data, features):
    """
    :param data:
    :return:
    """
    for user, df in data.items():
        f_scores = []
        for _, row in df.iterrows():
            f_values = [row[f_name] for f_name in features]
            f_scores.append(get_f(np.array(f_values)))
        # f_scores = [get_f(np.array([row.f1, row.f2])) for _, row in df.iterrows()]
        data[user]["F"] = pd.Series(f_scores, index=df.index)
    return data


def get_email(file_path, features):
    """
    :param file_path:
    :param features:
    :return:
    """
    pattern = '%m/%d/%Y %H:%M:%S'
    columns = ["epoch"] + features
    df = pd.read_csv(file_path)
    df = df.where((pd.notnull(df)), None)
    df = df[df.activity == "Send"]                      # Only keep emails which are sent (remove view and receive)
    users = df.user.unique()                            # Get a list of unique user names
    # Set up data dict
    data = {user: pd.DataFrame(columns=columns) for user in users}
    # Get frequency of each email address
    freq1, freq2 = get_email_freq(df)
    # Get the sum of email frequencies
    n1 = sum([value for value in freq1.values()])
    n2 = sum([value for value in freq2.values()])
    for _, row in df.iterrows():
        # Calculate frequency feature, f1
        f1 = get_f1(freq1, row, n1)
        # Calculate frequency feature, f2
        f2 = get_f2(freq2, row, n2)
        # Get UNIX time
        f3 = row["size"]
        epoch = int(time.mktime(time.strptime(row.date, pattern)))
        # Make a temporary data frame to append to the main data frame
        tmp = pd.DataFrame([[epoch, f1, f2, f3]], columns=columns)
        data[row.user] = data[row.user].append(tmp)
    return data


if __name__ == "__main__":
    features = ["f1", "f2", "f3"]
    data = get_email("../data/haystack_email.csv", features)
    data = sort_data(data)
    data = normalise_data(data, columns=features)
    data = calculate_combined_feature(data, features)
    n = [df.shape[0] for df in data.values()]
    print(len([i for i in n if i > 9]))

    i = 0
    for user, d in data.items():
        # plt.plot(d.epoch, d.freq)
        print(d)
        i += 1
        if i == 50:
            break
    # plt.show()



