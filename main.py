import argparse
import pandas as pd
import csv
import yaml
from .flags.filepath import *

from utils import *

file_paths = ["data/haystack_psychometric.csv"]

user_ids = get_unique_values(file_paths, "user_id", sep=",")

print(user_ids)


psychometrics = pd.read_csv(SC1_PSYCHO)

users = {}
for index, row in psychometrics.iterrows():
    users[row["user_id"]] = {"employee_name": row["employee_name"],
                             "pyschometric": {"openness": row["O"],
                                              "conscientiousness": row["C"],
                                              "extraversion": row["E"],
                                              "agreeableness": row["A"],
                                              "neuroticism": row["N"]}}

with open("test.yaml", 'w') as f:
    yaml.dump(users, f)