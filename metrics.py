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

    def set_decision_tree_time(self, time):
        self.decisionTreePerf = time

    def set_nca_knn_time(self, time):
        self.nca_knnPerf = time

    def set_random_forest_time(self, time):
        self.random_forestPerf = time

    def set_gaussian_score(self, score):
        self.gaussianScore = score

    def set_svm_score(self, score):
        self.svmScore = score

    def set_decision_tree_score(self, score):
        self.decisionTreeScore = score

    def set_nca_knn_score(self, score):
        self.nca_knnScore = score

    def set_random_forest_score(self, score):
        self.random_forestScore = score

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
              '\n*Random Forest: {}'\
            .format(self.route, self.windowSize, self.step_size, self.test_size,
                    'Yes' if self.useUnknown else 'No', self.xyPerf, self.windowPerf, self.featureExtractionPerf,
                    self.gaussianPerf, self.svmPerf, self.decisionTreePerf, self.nca_knnPerf, self.random_forestPerf, totalTime,
                    self.gaussianScore, self.svmScore, self.decisionTreeScore,
                    self.nca_knnScore, self.random_forestScore)

        output_loc = 'metrics/{}_{}.log'.format(self.route, datetime.now().strftime('%Y%m%d_%H%M%S'))

        with open(output_loc, 'w+') as f:
            f.write(out)

        print('\nWrote metrics to: {}\n'.format(output_loc))
