import os
import cv2
import random
import numpy as np
import mxnet as mx
from mxnet import nd
from PIL import Image
from scipy.spatial.transform import Rotation as ROT


class ShapeNetDataLoader(mx.io.DataIter):
    def __init__(self, ctx, opt):
        super(ShapeNetDataLoader, self).__init__()
        self.ctx = ctx
        self.opt = opt
        self.batch_size = opt.batchSize
        self.category = opt.category
        self.data_root = 'F:/Datasets/continuous_view_synthesis_datasets/dataset_%s' % opt.category

        # (Batch Size, Channels, Height, Width)
        self.data_shapes = (opt.batchSize, 3, self.opt.image_height, self.opt.image_width)
        self.label_shapes = (opt.batchSize, 3, self.opt.image_height, self.opt.image_width)
        self.poses_shapes = (opt.batchSize, 4, 4)

        with open(os.path.join(self.data_root, 'id_train.txt'), 'r') as fp:
            ids_train = [s.strip() for s in fp.readlines() if s]

        self.ids = ids_train
        self.opt.bound = 2
        self.num_examples = len(self.ids)
        self.num_batches = np.floor(self.num_examples / self.batch_size)

        self.cur_batch = 0
        self.reset()

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_examples

    def __next__(self):
        return self.next()

    def reset(self):
        random.shuffle(self.ids)
        self.cur_batch = 0

    def next(self):
        if self.cur_batch < self.num_batches:
            data = np.zeros(self.data_shapes, dtype=np.float32)
            label = np.zeros(self.label_shapes, dtype=np.float32)
            poses = np.zeros(self.poses_shapes, dtype=np.float32)

            for i in range(self.batch_size):
                idx_model = self.ids[self.cur_batch * self.batch_size + i]
                elev_a = 20 if not self.opt.random_elevation else random.randint(0, 2) * 10
                azim_a = random.randint(0, 17) * 20
                id_a = '%s_%d_%d' % (idx_model, azim_a, elev_a)

                elev_b = 20 if not self.opt.random_elevation else random.randint(0, 2) * 10
                delta = random.choice([20 * x for x in range(-self.opt.bound, self.opt.bound + 1) if x != 0])
                azim_b = (azim_a + delta) % 360
                id_b = '%s_%d_%d' % (idx_model, azim_b, elev_b)

                A = self.load_image(id_a) / 255. * 2 - 1
                B = self.load_image(id_b) / 255. * 2 - 1

                # Transpose image from (Height, Width, Channels) to (Channels, Height, Width)
                A = np.array(A.astype(np.float32)).transpose((2, 0, 1))
                B = np.array(B.astype(np.float32)).transpose((2, 0, 1))

                T = np.array([0, 0, 2]).reshape((3, 1))

                RA = ROT.from_euler('xyz', [-elev_a, azim_a, 0], degrees=True).as_matrix()
                RB = ROT.from_euler('xyz', [-elev_b, azim_b, 0], degrees=True).as_matrix()
                R = RA.T @ RB

                T = -R.dot(T) + T
                mat = np.block([[R, T],
                                [np.zeros((1, 3)), 1]])

                data[i, :, : ,:] = A[:, : ,:]
                label[i, :, : ,:] = B[:, : ,:]
                poses[i, :, :] = mat.astype(np.float32)[:, :]

            self.cur_batch += 1
            return mx.nd.array(data, ctx=self.ctx), mx.nd.array(label, ctx=self.ctx), mx.nd.array(poses, ctx=self.ctx)
        else:
            self.reset()
            raise StopIteration

    def load_image(self, id):
        image_path = os.path.join(self.data_root, 'images', id + '.png')
        pil_image = Image.open(image_path)
        image = np.asarray(pil_image.convert('RGB'))
        image = cv2.resize(image, (self.opt.image_width, self.opt.image_height))
        pil_image.close()
        return image


if __name__ == "__main__":
    from options import train_options
    opt = train_options.TrainOptions()
    opt = opt.parse()
    data_loader = ShapeNetDataLoader(mx.gpu(0), opt)
    data, label, pose = data_loader.next()
    print(data)
    print(label)
    print(pose)