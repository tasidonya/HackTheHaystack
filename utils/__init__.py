import pandas


def get_unique_values(file_paths, col_name, **kwargs):
    """
    :param file_paths: list: list of file paths to look in
    :param col_name: the column name to get unique values for
    :param kwargs: any kwargs for pandas.DataFrame() e.g. sep=","
    :return: numpy.ndarray
    """
    df = pandas.DataFrame(columns=[col_name])
    for path in file_paths:
        tmp = pandas.read_csv(path, **kwargs)
        df = df.append(tmp, ignore_index=True, sort=False)
    return df[col_name].unique()


def get_activities(file_path, columns=["activity", "pc", "date"]):
    """
    :param file_path: str: path to the file
    :param columns: list of str: headers to be included in activities
    :return: dict: data
    """
    df = pandas.read_csv(file_path)
    df = df.head()
    # Initialise the dictionary
    data = {}
    # Iterate through the rows
    for _, row in df.iterrows():
        user = row["user"]
        # If user is not in data, add user to data
        if user not in data:
            data.update({user: {"activities": []}})
        # For each item in columns, get a dictionary of the data for the user
        d = {item: row[item] for item in columns}
        # Append the data to the user activities
        data[user]["activities"].append(d)
    return data


if __name__ == "__main__":
    # print(get_unique_values(["data/1.csv", "data/2.csv"], "user_id", sep=","))
    print(get_activities("../data/haystack_file.csv"))


