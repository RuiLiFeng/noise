import tensorflow as tf
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from training import misc
import os
from metrics import metric_base
from metrics.metric_defaults import metric_defaults


import re
import sys

import pretrained_networks


def read_image(img):
    I = PIL.Image.open(img)
    I = I.convert('RGB')
    I = (np.array(I, np.float32) / 255.0 - 0.5) * 2
    if I.shape[-1] == 3:
        I = I.transpose([2, 0, 1])
    return I


def read_images(src_dir):
    imgs = []
    for i,j,k in os.walk(src_dir):
        for e in k:
            if e.endswith('.png'):
                imgs.append(read_image(os.path.join(i, e)))
    return imgs


def scale_alpha(alpha, t):
    return np.power(alpha, t) / (np.power(alpha, t) + np.power(1-alpha, t))


def embed(batch_size, resolution, imgs, network, iteration, result_dir, seed=6600):
    tf.reset_default_graph()
    print('Loading networks from "%s"...' % network)
    tflib.init_tf()
    _, _, G = pretrained_networks.load_networks(network)
    img_in = tf.placeholder(tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    noise_vars = [var for name, var in G.components.synthesis.vars.items() if name.startswith('noise')]
    alpha_vars = [var for name, var in G.components.synthesis.vars.items() if name.endswith('alpha')]
    alpha_eval = [alpha.eval() for alpha in alpha_vars]

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.randomize_noise = False
    G_syn = G.components.synthesis

    rnd = np.random.RandomState(seed)
    dlatent_avg = [var for name, var in G.vars.items() if name.startswith('dlatent_avg')][0].eval()
    dlatent_avg = np.expand_dims(np.expand_dims(dlatent_avg, 0), 1)
    dlatent_avg = dlatent_avg.repeat(12, 1)
    dlatent = tf.get_variable('dlatent', dtype=tf.float32, initializer=tf.constant(dlatent_avg),
                              trainable=True)
    synth_img = G_syn.get_output_for(dlatent, is_training=False, **G_kwargs)
    # synth_img = (synth_img + 1.0) / 2.0

    with tf.variable_scope('mse_loss'):
        mse_loss = tf.reduce_mean(tf.square(img_in - synth_img))
    with tf.variable_scope('perceptual_loss'):
        vgg_in = tf.concat([img_in, synth_img], 0)
        tf.keras.backend.set_image_data_format('channels_first')
        vgg = tf.keras.applications.VGG16(include_top=False, input_tensor=vgg_in, input_shape=(3, 128, 128),
                                          weights='/gdata2/fengrl/metrics/vgg.h5',
                                          pooling=None)
        h1 = vgg.get_layer('block1_conv1').output
        h2 = vgg.get_layer('block1_conv2').output
        h3 = vgg.get_layer('block3_conv2').output
        h4 = vgg.get_layer('block4_conv2').output
        pcep_loss = tf.reduce_mean(tf.square(h1[0] - h1[1])) + tf.reduce_mean(tf.square(h2[0] - h2[1])) + \
                    tf.reduce_mean(tf.square(h3[0] - h3[1])) + tf.reduce_mean(tf.square(h4[0] - h4[1]))
    loss = 0.5 * mse_loss + 0.5 * pcep_loss
    with tf.control_dependencies([loss]):
        train_op = opt.minimize(loss, var_list=[dlatent])
    reset_opt = tf.variables_initializer(opt.variables())
    reset_dl = tf.variables_initializer([dlatent])

    tflib.init_uninitialized_vars()
    # rnd = np.random.RandomState(seed)
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    idx = 0
    metrics_l = []
    metrics_p = []
    metrics_m = []
    metrics_d = []

    metrics_args = [metric_defaults[x] for x in ['fid50k', 'ppl_wend']]

    metrics_fun = metric_base.MetricGroup(metrics_args)
    for temperature in [0.2, 0.5, 1.0, 1.5, 2.0, 10.0]:
        tflib.set_vars({alpha: scale_alpha(alpha_np, temperature) for alpha, alpha_np in zip(alpha_vars, alpha_eval)})
        # misc.save_pkl((G, G, G), os.path.join(result_dir, 'temp%f.pkl' % temperature))
        # metrics_fun.run(os.path.join(result_dir, 'temp%f.pkl' % temperature), run_dir=result_dir,
        #                 data_dir='/gdata/fengrl/noise_test_dset/tfrecords',
        #                 dataset_args=dnnlib.EasyDict(tfrecord_dir='ffhq-128', shuffle_mb=0),
        #                 mirror_augment=True, num_gpus=1)
        for img in imgs:
            img = np.expand_dims(img, 0)
            loss_list = []
            p_loss_list = []
            m_loss_list = []
            dl_list = []
            si_list = []
            tflib.run([reset_opt, reset_dl])
            for i in range(iteration):
                loss_, p_loss_, m_loss_, dl_, si_, _ = tflib.run([loss, pcep_loss, mse_loss, dlatent, synth_img, train_op],
                                                                 {img_in: img})
                loss_list.append(loss_)
                p_loss_list.append(p_loss_)
                m_loss_list.append(m_loss_)
                dl_loss_ = np.sum(np.square(dl_-dlatent_avg))
                dl_list.append(dl_loss_)
                if i % 500 == 0:
                    si_list.append(si_)
                if i % 100 == 0:
                    print('Temperature %f, idx %d, Loss %f, mse %f, ppl %f, dl %f, step %d' %
                          (temperature, idx, loss_, m_loss_, p_loss_, dl_loss_, i))
            print('Temperature %f, idx %d, loss: %f, ppl: %f, mse: %f, d: %f' %
                  (temperature,
                   idx,
                   loss_list[-1],
                   p_loss_list[-1],
                   m_loss_list[-1],
                   dl_list[-1]))
            metrics_l.append(loss_list[-1])
            metrics_p.append(p_loss_list[-1])
            metrics_m.append(m_loss_list[-1])
            metrics_d.append(dl_list[-1])
            misc.save_image_grid(np.concatenate(si_list, 0), os.path.join(result_dir, 'temp%fsi%d.png' % (temperature, idx)), drange=[-1, 1])
            misc.save_image_grid(si_list[-1], os.path.join(result_dir, 'temp%fsifinal%d.png' % (temperature, idx)),
                                 drange=[-1, 1])
            with open(os.path.join(result_dir, 'temp%fmetric_l%d.txt' % (temperature, idx)), 'w') as f:
                for l_ in loss_list:
                    f.write(str(l_) + '\n')
            with open(os.path.join(result_dir, 'temp%fmetric_p%d.txt' % (temperature, idx)), 'w') as f:
                for l_ in p_loss_list:
                    f.write(str(l_) + '\n')
            with open(os.path.join(result_dir, 'temp%fmetric_m%d.txt' % (temperature, idx)), 'w') as f:
                for l_ in m_loss_list:
                    f.write(str(l_) + '\n')
            with open(os.path.join(result_dir, 'temp%fmetric_d%d.txt' % (temperature, idx)), 'w') as f:
                for l_ in dl_list:
                    f.write(str(l_) + '\n')
            idx += 1

        l_mean = np.mean(metrics_l)
        p_mean = np.mean(metrics_p)
        m_mean = np.mean(metrics_m)
        d_mean = np.mean(metrics_d)
        print('Overall metrics: temp %f, loss_mean %f, ppl_mean %f, mse_mean %f, d_mean %f' % (temperature, l_mean, p_mean, m_mean, d_mean))
        with open(os.path.join(result_dir, 'mean_metrics'), 'a') as f:
            f.write('Temperature %f\n' % temperature)
            f.write('loss %f\n' % l_mean)
            f.write('mse %f\n' % m_mean)
            f.write('ppl %f\n' % p_mean)
            f.write('dl %f\n' % d_mean)


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--resolution', default=128, type=int)
    parser.add_argument('--src_dir', default="/gdata2/fengrl/imgs-for-embed/")
    parser.add_argument('--network', default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt", type=str)
    parser.add_argument('--iteration', default=1000, type=int)
    parser.add_argument('--result_dir', default='/gdata2/fengrl/inverse')

    args = parser.parse_args()

    # vgg = misc.load_pkl('/gdata2/fengrl/metrics/vgg16_zhang_perceptual.pkl')

    imgs = read_images(args.src_dir)

    embed(args.batch_size, args.resolution, imgs, args.network, args.iteration, args.result_dir)


if __name__ == "__main__":
    main()
