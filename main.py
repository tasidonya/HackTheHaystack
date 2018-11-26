import argparse

from utils import *

file_paths = ["data/haystack_psychometric.csv"]

user_ids = get_unique_values(file_paths, "user_id", sep=",")

print(user_ids)