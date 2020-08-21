import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd
from options.train_options import TrainOptions
from data.shapenet_data_loader import ShapeNetDataLoader
from util.visualizer import Visualizer
from models.network import Network
from util import util
from util.ranger_optimizer import Ranger

np.random.seed(1)
mx.random.seed(1)

opt = TrainOptions().parse()
ctx = mx.gpu(0)
train_iter = ShapeNetDataLoader(ctx, opt)
dataset_size = len(train_iter)
print('#training samples = %d' % dataset_size)

visualizer = Visualizer(opt)
save_dir = os.path.join(opt.checkpoints_dir, opt.name)

depth_bias, depth_scale = 0, 1.
intrinsics = nd.zeros((1, 3, 3), ctx=ctx, dtype=np.float32).reshape((1, 3, 3))
if opt.category == 'kitti':
    intrinsics = np.array(
        [718.9, 0., 128,
         0., 718.9, 128,
         0., 0., 1.]).reshape((3, 3))
    depth_bias, depth_scale = 0, 1.
    depth_scale_vis = 250. / depth_scale
    depth_bias_vis = 0.
    intrinsics = nd.array(intrinsics, ctx=ctx, dtype=np.float32).reshape((1, 3, 3))
elif opt.category in ['car','chair']:
    intrinsics = np.array([480, 0, 128,
                           0, 480, 128,
                           0, 0, 1]).reshape((3, 3))
    depth_bias, depth_scale = 2, 1.
    depth_scale_vis = 125. / depth_scale
    depth_bias_vis = depth_bias - depth_scale
    intrinsics = nd.array(intrinsics, ctx=ctx, dtype=np.float32).reshape((1, 3, 3))

net = Network(opt, depth_scale, depth_bias)
loss_function = gluon.loss.L1Loss()

# net.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
net.initialize(mx.init.Normal(0.02), ctx=ctx)

trainer = gluon.Trainer(net.collect_params(), 'nadam', {'learning_rate': opt.lr, 'wd': opt.wd, 'beta1': opt.beta1})
# optimizer = Ranger(learning_rate=opt.lr, wd=opt.wd, alpha=0.75, beta1=opt.beta1, k=6, n_sma_threshhold=4,
#                                     use_gc=False, gc_conv_only=True)
# trainer = gluon.Trainer(net.collect_params(), optimizer)

lr_decay_count = 0
lr_decay = 0.1
lr_decay_epoch = [30, np.inf] # [20, 30, np.inf]

smoothing_constant = .01
moving_loss = 0
total_steps = 0

for epoch in range(opt.epoch_count, opt.num_epochs):
    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        lr_decay_count += 1
        print("Update Learning Rate: " + str(trainer.learning_rate * lr_decay))

    epoch_start_time = time.time()
    epoch_iter = 0

    iter_start_time = 0
    for i, (data, label, pose) in enumerate(train_iter):
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # Train
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        pose = pose.as_in_context(ctx)
        with autograd.record():
            output, flow, mask = net(data, pose, intrinsics)
            loss = loss_function(output, label) * opt.lambda_recon
            loss.backward()
        trainer.step(data.shape[0])

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (i == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(util.get_current_visuals(data, label, output), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = {'loss_G': curr_loss}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, dataset_size, errors, t)
            # if opt.display_id > 0:
            #     visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

        iter_start_time = time.time()

    visualizer.display_current_anim(util.get_current_anim(opt, ctx, net, data, intrinsics), epoch)

    print('saving the model at the end of epoch %d, iters %d' %
          (epoch, total_steps))
    save_filename = 'net_latest.params'
    save_path = os.path.join(save_dir, save_filename)
    net.save_parameters(save_path)
    if epoch % opt.save_epoch_freq == 0:
        save_filename = 'net_%s.params' % epoch
        save_path = os.path.join(save_dir, save_filename)
        net.save_parameters(save_path)

    print('End of epoch %d / %d \t Time Taken: %d sec \t Loss: %.8f' %
          (epoch, opt.num_epochs, time.time() - epoch_start_time, moving_loss))
