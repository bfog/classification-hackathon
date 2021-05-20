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

        self.gaussianScore = ''
        self.svmScore = ''

    def set_xy(self, time):
        self.xyPerf = time

    def set_window(self, time):
        self.windowPerf = time

    def set_feature_extraction(self, time):
        self.featureExtractionPerf = time

    def set_gaussian_time(self, time):
        self.gaussianPerf = time

    def set_svm_time(self, time):
        self.svmPerf = time

    def set_gaussian_score(self, score):
        self.gaussianScore = score

    def set_svm_score(self, score):
        self.svmScore = score

    def write(self):
        if not os.path.exists('metrics'):
            os.mkdir('metrics')

        totalTime = self.xyPerf + self.windowPerf + self.featureExtractionPerf + self.gaussianPerf + self.svmPerf
        out = 'Route used: {}\n' \
              'Window size: {}\n' \
              'Step size: {}\n' \
              'Test split size: {}\n' \
              'Included unknowns: {}\n\n' \
              'Classification metrics:\n' \
              '\n*Xy preprocessing: {}s\n*Window preprocessing: {}s\n*Feature Extraction: {}s' \
              '\n*Gaussian Training: {}s\n*SVM Training: {}s\n\nTotal time: {}s' \
              '\n\nML Model scores:' \
              '\n*Gaussian NB: {}' \
              '\n*SVM: {}'\
            .format(self.route, self.windowSize, self.step_size, self.test_size,
                    'Yes' if self.useUnknown else 'No', self.xyPerf, self.windowPerf, self.featureExtractionPerf,
                    self.gaussianPerf, self.svmPerf, totalTime, self.gaussianScore, self.svmScore)

        output_loc = 'metrics/{}_{}.log'.format(self.route, datetime.now().strftime('%Y%m%d_%H%M%S'))

        with open(output_loc, 'w+') as f:
            f.write(out)

        print('Wrote metrics to: {}\n'.format(output_loc))
