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
        df = df.append(tmp, ignore_index=True)
    return df[col_name].unique()


if __name__ == "__main__":
    print(get_unique_values(["data/1.csv", "data/2.csv"], "user_id", sep=","))
