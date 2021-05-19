import data_set
import metrics

import argparse


def run(args):
    appMetrics = metrics.Metrics()

    data = data_set.DcsData(args, appMetrics)
    data.run()

    appMetrics.write()


# TODO: Add option to use external preprocessed data (e.g. Aj's data)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML hackathon')
    parser.add_argument('-d', '--data', choices=['all', 'Route1', 'Route2', 'Route3', 'Route4'],
                        default='all', help='Which training data to use')
    parser.add_argument('-u', '--unknown', help='Use unknown data', action='store_true')
    parser.add_argument('-r', '--rerun', help='Rerun data preprocessing', action='store_true')
    parser.add_argument('-w', '--window_size', type=int, default=300, help='Sliding window size')
    parser.add_argument('-s', '--step_size', type=int, default=100, help='Step size')

    run(parser.parse_args())


