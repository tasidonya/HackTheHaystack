import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


class EmailFeatures(object):

    def __init__(self, file_path):
        self.features = ["f1", "f2", "f3"]
        self.data = None
        self.file_path = file_path
        self.get_email()
        self.sort_data()
        self.normalise_data()
        self.calculate_combined_feature()

    def get_email(self):
        print("Fetching data and computing features...")
        pattern = '%m/%d/%Y %H:%M:%S'
        columns = ["epoch"] + self.features
        df = pd.read_csv(self.file_path)
        df = df.where((pd.notnull(df)), None)
        df = df[df.activity == "Send"]  # Only keep emails which are sent (remove view and receive)
        users = df.user.unique()  # Get a list of unique user names
        # Set up data dict
        data = {user: pd.DataFrame(columns=columns) for user in users}
        # Get frequency of each email address
        freq1, freq2 = self.get_email_freq(df)
        # Get the sum of email frequencies
        n1 = sum([value for value in freq1.values()])
        n2 = sum([value for value in freq2.values()])
        for _, row in df.iterrows():
            # Calculate frequency feature, f1
            f1 = self.get_f1(freq1, row, n1)
            # Calculate frequency feature, f2
            f2 = self.get_f2(freq2, row, n2)
            # Get UNIX time
            f3 = row["size"]
            epoch = int(time.mktime(time.strptime(row.date, pattern)))
            # Make a temporary data frame to append to the main data frame
            tmp = pd.DataFrame([[epoch, f1, f2, f3]], columns=columns)
            data[row.user] = data[row.user].append(tmp)
        self.data = data
        print("Done")

    def sort_data(self):
        """
        :return:
        """
        print("Sorting Data...")
        for user, item in self.data.items():
            self.data[user] = item.sort_values(by=["epoch"])
        print("Done")

    def normalise_data(self):
        """
        :return:
        """
        print("Normalising Data...")
        maximum = [0 for _ in self.features]
        for user, df in self.data.items():
            for i, (col, m) in enumerate(zip(self.features, maximum)):
                col_max = df[col].max()
                if col_max > m:
                    maximum[i] = col_max
        for user, df in self.data.items():
            for col, m in zip(self.features, maximum):
                df[col] /= m
        print("Done")

    @staticmethod
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

    @staticmethod
    def get_f1(freq, row, n):
        """
        :param freq:
        :param row:
        :param n:
        :return:
        """
        f = np.asarray([freq[address] for address in row.to.split(";")])
        return np.max(- np.log(f / n))

    @staticmethod
    def get_f2(freq, row, n):
        """
        :param freq:
        :param row:
        :param n:
        :return:
        """
        f = np.asarray([freq[address.split("@")[1]] for address in row.to.split(";")])
        return np.max(- np.log(f / n))

    @staticmethod
    def get_f(f_values):
        """
        :param f_values:
        :return:
        """
        return (np.prod(f_values) * np.sum(f_values)) / len(f_values)

    def calculate_combined_feature(self):
        """
        :return:
        """
        print("Calculating combined feature")
        for user, df in self.data.items():
            f_scores = []
            for _, row in df.iterrows():
                f_values = [row[f_name] for f_name in self.features]
                f_scores.append(self.get_f(np.array(f_values)))
            self.data[user]["F"] = pd.Series(f_scores, index=df.index)
        print("Done")

    def get(self):
        return self.data


if __name__ == "__main__":
    email_features = EmailFeatures("../data/haystack_email.csv")
    data = email_features.get()

    n = [df.shape[0] for df in data.values()]
    print(len([i for i in n if i > 9]))
    input("")
    i = 0
    for user, d in data.items():
        # plt.plot(d.epoch, d.freq)
        print(d)
        i += 1
        if i == 50:
            break
    # plt.show()



