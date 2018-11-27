import pandas as pd
import numpy as np
import time
import re
import matplotlib.pyplot as plt


class UserName(object):

    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.user_dict = {row.user_id: row.employee_name for _, row in df.iterrows()}

    def get(self):
        return self.user_dict


class EmailFeatures(object):

    def __init__(self, file_path, nrows=None):
        self.features = ["f1", "f2", "f3", "f4", "f5"]
        self.data = None
        self.file_path = file_path
        self.get_email(nrows=nrows)
        self.sort_data()
        self.normalise_data()
        self.calculate_combined_feature()

    def get_email(self, nrows):
        print("Fetching data and computing features...")
        pattern = '%m/%d/%Y %H:%M:%S'
        columns = ["epoch"] + self.features
        df = pd.read_csv(self.file_path, nrows=nrows)
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
            # Get email size
            f3 = row["size"]
            # Get email address similarity score
            f4 = self.compute_similarity(row["from"], row["to"].split(";"))
            f5 = 1 / len(row.content) if len(row.content) > 0 else 1
            epoch = int(time.mktime(time.strptime(row.date, pattern)))
            # Make a temporary data frame to append to the main data frame
            tmp = pd.DataFrame([[epoch, f1, f2, f3, f4, f5]], columns=columns)
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

    # Cosine distance https://stackoverflow.com/questions/29484529/cosine-similarity-between-two-words-in-a-list
    @staticmethod
    def word2vec(word):
        from collections import Counter
        from math import sqrt

        # count the characters in word
        cw = Counter(word)
        # precomputes a set of the different characters
        sw = set(cw)
        # precomputes the "length" of the word vector
        lw = sqrt(sum(c * c for c in cw.values()))

        # return a tuple
        return cw, sw, lw

    # Cosine distance https://stackoverflow.com/questions/29484529/cosine-similarity-between-two-words-in-a-list
    @staticmethod
    def cosdis(v1, v2):
        # which characters are common to the two words?
        common = v1[1].intersection(v2[1])
        # by definition of cosine distance we have
        return sum(v1[0][ch] * v2[0][ch] for ch in common) / v1[2] / v2[2]

    # Extracts part before @
    @staticmethod
    def first_part_extract(email_list):
        first_parts = []
        for email in email_list:
            first_parts.append(re.match("(.*)@", email, flags=0).group(1))
        return first_parts

    def compute_similarity(self, e_from, to):
        first_part_to = re.match("(.*)@", e_from, flags=0).group(1)  # Extract just the part before @
        from_to_vec = self.word2vec(first_part_to)
        first_part_from = self.first_part_extract(to)
        # Turn it all to vec
        to_name_vecs = []
        for email in first_part_from:
            to_name_vecs.append(self.word2vec(email))

        cosdises = []
        for email in to_name_vecs:
            cosdises.append(self.cosdis(from_to_vec, email))

        cosdises.sort(reverse=True)
        return cosdises[0]

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
        print("Calculating combined feature...")
        for user, df in self.data.items():
            f_scores = []
            for _, row in df.iterrows():
                f_values = [row[f_name] for f_name in self.features]
                f_scores.append(self.get_f(np.array(f_values)))
            self.data[user]["F"] = pd.Series(f_scores, index=df.index)
        print("Done")

    def get(self):
        return self.data

    def get_for_training(self, n=5):
        """
        :param n:
        :return:
        """
        data = []
        for user, df in self.data.items():
            sample = np.zeros((n, len(self.features)), dtype=np.float32)
            if df.shape[0] >= n:
                for i, (_, row) in enumerate(df.head(n).iterrows()):
                    sample[i, :] = np.array([row[feature] for feature in self.features])
                data.append(sample)
        return data


if __name__ == "__main__":
    email_features = EmailFeatures("../data/haystack_email.csv", nrows=10000)
    data = email_features.get()
    i = 0
    for user, d in data.items():
        # plt.plot(d.epoch, d.freq)
        print(d)
        i += 1
        if i == 100:
            break
    # plt.show()



