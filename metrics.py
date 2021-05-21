import os
from datetime import datetime


class Metrics:
    def __init__(self, args):
        self.route = args.data
        self.windowSize = args.window_size
        self.step_size = args.step_size
        self.useUnknown = args.unknown
        self.test_size = args.test_size
        self.xyPerf = 0
        self.windowPerf = 0
        self.featureExtractionPerf = 0
        self.gaussianPerf = 0
        self.svmPerf = 0
        self.decisionTreePerf = 0
        self.nca_knnPerf = 0
        self.random_forestPerf = 0

        self.gaussianScore = ''
        self.svmScore = ''
        self.decisionTreeScore = ''
        self.nca_knnScore = ''
        self.random_forestScore = ''

        self.gridSearchMetrics = ''

    def write(self):
        if not os.path.exists('metrics'):
            os.mkdir('metrics')

        totalTime = self.xyPerf + self.windowPerf + self.featureExtractionPerf + self.gaussianPerf + self.svmPerf + self.decisionTreePerf + self.nca_knnPerf + self.random_forestPerf
        out = 'Route used: {}\n' \
              'Window size: {}\n' \
              'Step size: {}\n' \
              'Test split size: {}\n' \
              'Included unknowns: {}\n\n' \
              'Classification metrics:\n' \
              '\n*Xy preprocessing: {}s\n*Window preprocessing: {}s\n*Feature Extraction: {}s' \
              '\n*Gaussian Training: {}s\n*SVM Training: {}s\n*Decision Tree: {}\n*NCA_KNN: {}\n*Random Forest: {}s\n\nTotal time: {}s' \
              '\n\nML Model scores:' \
              '\n*Gaussian NB: {}' \
              '\n*SVM: {}' \
              '\n*Decision Tree: {}' \
              '\n*NCA & KNN: {}' \
              '\n*Random Forest: {}' \
              '\n{}'\
            .format(self.route, self.windowSize, self.step_size, self.test_size,
                    'Yes' if self.useUnknown else 'No', self.xyPerf, self.windowPerf, self.featureExtractionPerf,
                    self.gaussianPerf, self.svmPerf, self.decisionTreePerf, self.nca_knnPerf, self.random_forestPerf, totalTime,
                    self.gaussianScore, self.svmScore, self.decisionTreeScore,
                    self.nca_knnScore, self.random_forestScore, self.gridSearchMetrics)

        output_loc = 'metrics/{}_{}.log'.format(self.route, datetime.now().strftime('%Y%m%d_%H%M%S'))

        with open(output_loc, 'w+') as f:
            f.write(out)

        print('\nWrote metrics to: {}\n'.format(output_loc))
