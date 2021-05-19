import os
from datetime import datetime


class Metrics:
    def __init__(self):
        self.xyPerf = 0
        self.windowPerf = 0

    def set_xy(self, time):
        self.xyPerf = time

    def set_window(self, time):
        self.windowPerf = time

    def write(self):
        if not os.path.exists('metrics'):
            os.mkdir('metrics')
        out = 'Classification metrics:\nXy preprocessing: {}s\nWindow preprocessing:{}s'\
            .format(self.xyPerf, self.windowPerf)

        output_loc = 'metrics/{}.log'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))

        with open(output_loc, 'w+') as f:
            f.write(out)

        print('Wrote metrics to: {}\n'.format(output_loc))
