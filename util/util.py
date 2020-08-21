from __future__ import print_function
import os
import numpy as np
from mxnet import nd
from numpy.lib.stride_tricks import as_strided
from scipy.spatial.transform import Rotation as ROT
from collections import OrderedDict
from PIL import Image


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].asnumpy()
    if image_numpy.ndim == 2:
        image_numpy = image_numpy.reshape((1,image_numpy.shape[0],image_numpy.shape[1]))
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))

    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(imtype).copy()

    return image_numpy


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tile_array(a, b0, b1):
    r, c = a.shape                                    # number of rows/columns
    rs, cs = a.strides                                # row/column strides
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
    return x.reshape((r*b0, c*b1))


def get_current_visuals(real_A, real_B, fake_B):
    out_dict = {}
    out_dict['real_A'] = tensor2im(real_A)
    out_dict['real_B'] = tensor2im(real_B)
    out_dict['fake_B'] = tensor2im(fake_B)
    out_dict['diff'] = tensor2im(0.5*(real_B[0,:,:,:] - fake_B[0,:,:,:]).abs())
    return OrderedDict(out_dict)


def eval(network, real_A, real_B, real_RT, l1_loss, ssim_loss):
    fake_B, flow, mask = network(real_A, real_RT)
    out_dict = {}
    out_dict['L1'] = l1_loss(fake_B, real_B)
    out_dict['SSIM'] = ssim_loss((fake_B + 1) / 2, (real_B + 1) / 2)
    print(out_dict)
    exit()
    return OrderedDict(out_dict)


def get_current_anim(opt, ctx, network, real_A, intrinsics):
    anim_dict = {'vis': []}
    real_A = real_A[:1]

    NV = 60
    for i in range(NV):
        pose = np.array([0, -(i - NV / 2) * np.pi / 180, 0, 0, 0, 0]) if opt.category in ['car', 'chair'] \
            else np.array([0, 0, 0, 0, 0, i / 1000])
        real_RT = get_RT(opt, ctx, pose)
        fake_B, flow, mask = network(real_A, real_RT, intrinsics)
        anim_dict['vis'].append(tensor2im(fake_B))
    return anim_dict


def get_RT(opt, ctx, pose1):
    if opt.category in ['car', 'chair']:
        T = np.array([0, 0, 2]).reshape((3, 1))
        R = ROT.from_euler('xyz', pose1[:3]).as_matrix()
        T = -R.dot(T) + T
    else:
        R = ROT.from_euler('xyz', pose1[0:3]).as_matrix()
        T = pose1[3:].reshape((3, 1))
    mat = np.block(
        [[R, T],
         [np.zeros((1, 3)), 1]])
    return nd.array(mat, ctx=ctx, dtype=np.float32).reshape((1, 4, 4))