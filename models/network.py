import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from models.projection_layer import inverse_warp

class Encoder(gluon.HybridBlock):
    def __init__(self, nz=200, use_dropout=False):
        super(Encoder, self).__init__()
        self.use_dropout = use_dropout
        nc = 36 # num channels

        # Input: (batch, 3, 256, 256)
        # Output: (batch, 32, 128, 128)
        self.conv_0 = nn.Conv2D(channels=nc, kernel_size=(4, 4), strides=(2, 2), padding=(1, 1))
        self.conv_bn_0 = nn.BatchNorm()
        self.conv_act_0 = nn.PReLU()

        # Input: (batch, 32, 128, 128)
        # Output: (batch, 64, 64, 64)
        self.conv_1 = nn.Conv2D(channels=nc * 2, kernel_size=(4, 4), strides=(2, 2), padding=(1, 1))
        self.conv_bn_1 = nn.BatchNorm()
        self.conv_act_1 = nn.PReLU()

        # Input: (batch, 64, 64, 64)
        # Output: (batch, 128, 32, 32)
        self.conv_2 = nn.Conv2D(channels=nc * 4, kernel_size=(4, 4), strides=(2, 2), padding=(1, 1))
        self.conv_bn_2 = nn.BatchNorm()
        self.conv_act_2 = nn.PReLU()

        # Input: (batch, 128, 32, 32)
        # Output: (batch, 256, 16, 16)
        self.conv_3 = nn.Conv2D(channels=nc * 8, kernel_size=(4, 4), strides=(2, 2), padding=(1, 1))
        self.conv_bn_3 = nn.BatchNorm()
        self.conv_act_3 = nn.PReLU()

        # Input: (batch, 256, 16, 16)
        # Output: (batch, 512, 8, 8)
        self.conv_4 = nn.Conv2D(channels=nc * 16, kernel_size=(4, 4), strides=(2, 2), padding=(1, 1))
        self.conv_bn_4 = nn.BatchNorm()
        self.conv_act_4 = nn.PReLU()

        # Input: (batch, 512, 8, 8)
        # Output: (batch, 1024, 4, 4)
        self.conv_5 = nn.Conv2D(channels=nc * 32, kernel_size=(4, 4), strides=(2, 2), padding=(1, 1))
        self.conv_bn_5 = nn.BatchNorm()
        self.conv_act_5 = nn.PReLU()

        # Input: (batch, 1024, 4, 4)
        # Output: (batch, 2048, 1, 1)
        self.conv_6 = nn.Conv2D(channels=nc * 64, kernel_size=(4, 4), strides=(4, 4), padding=(0, 0))
        self.conv_bn_6 = nn.BatchNorm()
        self.conv_act_6 = nn.PReLU()

        # Input: (batch, 2048, 1, 1)
        # Output: (batch, 2048)
        # Output: (batch, nz)
        self.fc_0 = nn.Dense(nz)
        if self.use_dropout:
            self.drop_0 = nn.Dropout(0.3)

    def hybrid_forward(self, F, x):
        x = self.conv_0(x)
        x = self.conv_bn_0(x)
        x = self.conv_act_0(x)
        x = self.conv_1(x)
        x = self.conv_bn_1(x)
        x = self.conv_act_1(x)
        x = self.conv_2(x)
        x = self.conv_bn_2(x)
        x = self.conv_act_2(x)
        x = self.conv_3(x)
        x = self.conv_bn_3(x)
        x = self.conv_act_3(x)
        x = self.conv_4(x)
        x = self.conv_bn_4(x)
        x = self.conv_act_4(x)
        x = self.conv_5(x)
        x = self.conv_bn_5(x)
        x = self.conv_act_5(x)
        x = self.conv_6(x)
        x = self.conv_bn_6(x)
        x = self.conv_act_6(x)
        x = F.Flatten(x)
        x = self.fc_0(x)
        if self.use_dropout:
            x = self.drop_0(x)
        return x


class UpBlock(nn.HybridBlock):
    def __init__(self, channels, billinear=False):
        super(UpBlock, self).__init__()
        self.billinear = billinear
        if self.billinear:
            self.conv0 = nn.Conv2D(channels=channels, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1))
        else:
            self.upsampler = nn.Conv2DTranspose(channels=channels, kernel_size=4, strides=2,
                                                padding=1, use_bias=True)

    def hybrid_forward(self, F, x):
        if self.billinear:
            # mx.nd.contrib.BilinearResize2D()
            x = F.contrib.BilinearResize2D(data=x, height=x.shape[3] * 2, width=x.shape[2] * 2)
            x = self.conv0(x)
            return x
        else:
            x = self.upsampler(x)
            return x


class Decoder(gluon.HybridBlock):
    def __init__(self, use_dropout=False):
        super(Decoder, self).__init__()
        self.use_dropout = use_dropout
        nc = 36

        # Input: (batch, nz_geo * 3)
        # Output: (batch, 2048)
        self.fc_0 = nn.Dense(2048)
        if self.use_dropout:
            self.drop_0 = nn.Dropout(0.3)
        self.fc_act_0 = nn.PReLU()

        # Input: (batch, 2048)
        # Input: (batch, 2048, 1, 1)
        # Output: (batch, 1024, 4, 4)
        self.conv_0 = nn.Conv2DTranspose(channels=nc * 32, kernel_size=4, strides=4, padding=0, use_bias=True)
        self.conv_bn_0 = nn.BatchNorm()
        self.conv_act_0 = nn.PReLU()

        # Input: (batch, 1024, 4, 4)
        # Output: (batch, 512, 8, 8)
        self.conv_1 = nn.Conv2DTranspose(channels=nc * 16, kernel_size=4, strides=2, padding=1, use_bias=True)
        self.conv_bn_1 = nn.BatchNorm()
        self.conv_act_1 = nn.PReLU()

        # Input: (batch, 512, 8, 8)
        # Output: (batch, 256, 16, 16)
        self.conv_2 = nn.Conv2DTranspose(channels=nc * 8, kernel_size=4, strides=2, padding=1, use_bias=True)
        self.conv_bn_2 = nn.BatchNorm()
        self.conv_act_2 = nn.PReLU()

        # Input: (batch, 256, 16, 16)
        # Output: (batch, 128, 32, 32)
        self.conv_3 = nn.Conv2DTranspose(channels=nc * 4, kernel_size=4, strides=2, padding=1, use_bias=True)
        self.conv_bn_3 = nn.BatchNorm()
        self.conv_act_3 = nn.PReLU()

        # Input: (batch, 128, 32, 32)
        # Output: (batch, 64, 64, 64)
        self.conv_4 = nn.Conv2DTranspose(channels=nc * 2, kernel_size=4, strides=2, padding=1, use_bias=True)
        self.conv_bn_4 = nn.BatchNorm()
        self.conv_act_4 = nn.PReLU()

        # Input: (batch, 64, 64, 64)
        # Output: (batch, 32, 128, 128)
        self.conv_5 = nn.Conv2DTranspose(channels=nc, kernel_size=4, strides=2, padding=1, use_bias=True)
        self.conv_bn_5 = nn.BatchNorm()
        self.conv_act_5 = nn.PReLU()

        # Input: (batch, 32, 128, 128
        # Output: (batch, 16, 256, 256)
        self.conv_6 = nn.Conv2DTranspose(channels=int(nc / 2), kernel_size=4, strides=2, padding=1, use_bias=True)
        self.conv_bn_6 = nn.BatchNorm()
        self.conv_act_6 = nn.PReLU()

        # Input: (batch, 16, 256, 256)
        # Output: (batch, 1, 256, 256)
        self.conv_7 = nn.Conv2D(channels=1, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0))

    def hybrid_forward(self, F, x):
        x = self.fc_0(x)
        if self.use_dropout:
            x = self.drop_0(x)
        x = self.fc_act_0(x)
        x = F.reshape(x, (x.shape[0], x.shape[1], 1, 1))

        x = self.conv_0(x)
        x = self.conv_bn_0(x)
        x = self.conv_act_0(x)
        x = self.conv_1(x)
        x = self.conv_bn_1(x)
        x = self.conv_act_1(x)
        x = self.conv_2(x)
        x = self.conv_bn_2(x)
        x = self.conv_act_2(x)
        x = self.conv_3(x)
        x = self.conv_bn_3(x)
        x = self.conv_act_3(x)
        x = self.conv_4(x)
        x = self.conv_bn_4(x)
        x = self.conv_act_4(x)
        x = self.conv_5(x)
        x = self.conv_bn_5(x)
        x = self.conv_act_5(x)
        x = self.conv_6(x)
        x = self.conv_bn_6(x)
        x = self.conv_act_6(x)
        x = self.conv_7(x)
        return x

class Network(gluon.HybridBlock):
    def __init__(self, opt, depth_scale, depth_bias):
        super(Network, self).__init__()
        self.opt = opt
        self.depth_scale = depth_scale
        self.depth_bias = depth_bias

        # Input: (batch, 3, 256, 256)
        # Output: (batch, nz_geo * 3)
        self.encoder = Encoder(opt.nz_geo * 3)
        # Input: (batch, nz_geo * 3)
        # Output: (batch, 1, 256, 256)
        self.decoder = Decoder()

    def hybrid_forward(self, F, x, poses, intrinsics):
        z = self.encoder(x)
        z_tf = self.transform(F, z, poses)
        depth = self.decode_z(F, z_tf)
        fake_B, flow, mask = inverse_warp(x, depth, poses, intrinsics)
        return fake_B, flow, mask

    def transform(self, F, z, RT):
        return self.transform_code(F, z, self.opt.nz_geo, F.linalg.inverse(RT),
                                   object_centric=self.opt.category in ['car', 'chair'])

    def transform_code(self, F, z, nz, RT, object_centric=False):
        z_tf = F.linalg.gemm2(z.reshape((z.shape[0], nz, 3)), RT[:, :3, :3])
        if not object_centric:
            z_tf = z_tf + RT[:, :3, 3].expand_dims(1).broadcast_to((RT.shape[0], nz, 3))
        return z_tf.reshape((z.shape[0], nz * 3))

    def decode_z(self, F, z):
        output = self.decoder(z)
        if self.opt.category in ['kitti']:
            return 1 / (10 * F.sigmoid(output) + 0.01) # predict disparity instead of depth for natural scenes
        elif self.opt.category in ['car', 'chair']:
            return F.tanh(output) * self.depth_scale + self.depth_bias
