import argparse
import os
from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.isTrain = True
        self.opt = None

    def initialize(self):
        # basic info
        self.parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        # visualization related
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')

        # data loading related
        self.parser.add_argument('--image_width', default=256, type=int, help='Image width')
        self.parser.add_argument('--image_height', default=256, type=int, help='Image height')

        # experiment related
        self.parser.add_argument('--nz_geo', type=int, default=256, help='number of latent points')
        self.parser.add_argument('--category', type=str, default='car', help='model category [car|chair|kitti]')
        self.parser.add_argument('--random_elevation', action='store_true', help='introduce elevation changes between training pairs')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
