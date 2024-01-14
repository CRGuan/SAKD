import os

class Path(object):

    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/home/test/dataset/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/home/test/dataset/sbd/benchmark_RELEASE/'  # folder that contains dataset/.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
