import data_set

import argparse


def run(args):
    data = data_set.DcsData(args)
    data.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML hackathon')
    parser.add_argument('-d', '--data', choices=['all', 'Route1', 'Route2', 'Route3', 'Route4'],
                        default='all', help='Which training data to use')
    parser.add_argument('-u', '--unknown', help='Use unknown data', action='store_true')
    parser.add_argument('-r', '--rerun', help='Rerun data preprocessing', action='store_true')

    run(parser.parse_args())

