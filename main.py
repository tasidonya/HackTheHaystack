import argparse
import matplotlib.pyplot as plt

from utils import EmailFeatures


def main(args):
    email_features = EmailFeatures(args.path, nrows=int(args.nrows))
    data = email_features.get()
    for user, d in data.items():
        if d.F.max() > args.threshold:
            plt.plot(d.epoch, d.F, label=user)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="path to email csv path")
    parser.add_argument("-n", "--nrows", default=None, help="Number of rows to load from the email csv file")
    parser.add_argument("-t", "--threshold", default=0.02, type=float, help="Threshold speciousness")
    main(parser.parse_args())
