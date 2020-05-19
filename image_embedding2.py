import tensorflow as tf
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from training import misc
import os

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


def embed(batch_size, resolution, imgs, network, iteration, result_dir, seed=6600):
    tf.reset_default_graph()
    print('Loading networks from "%s"...' % network)
    tflib.init_tf()
    _, _, G = pretrained_networks.load_networks(network)
    img_in = tf.placeholder(tf.float32)
    step = tf.get_variable('step', tf.float32, tf.constant(0.0))
    lr = tf.train.exponential_decay(0.01, global_step=step, decay_steps=3500, decay_rate=0.7)
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    noise_vars = [var for name, var in G.components.synthesis.vars.items() if name.startswith('noise')]

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
    with tf.control_dependencies([loss, tf.assign_add(lr, 1.0)]):
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
                print('idx %d, Loss %f, mse %f, ppl %f, dl %f, step %d' % (idx, loss_, m_loss_, p_loss_, dl_loss_, i))
        print('loss: %f, ppl: %f, mse: %f, d: %f' % (loss_list[-1],
                                                     p_loss_list[-1],
                                                     m_loss_list[-1],
                                                     dl_list[-1]))
        metrics_l.append(loss_list[-1])
        metrics_p.append(p_loss_list[-1])
        metrics_m.append(m_loss_list[-1])
        metrics_d.append(dl_list[-1])
        misc.save_image_grid(np.concatenate(si_list, 0), os.path.join(result_dir, 'si%d.png' % idx), drange=[-1, 1])
        misc.save_image_grid(si_list[-1], os.path.join(result_dir, 'sifinal%d.png' % idx),
                             drange=[-1, 1])
        with open(os.path.join(result_dir, 'metric_l%d.txt' % idx), 'w') as f:
            for l_ in loss_list:
                f.write(str(l_) + '\n')
        with open(os.path.join(result_dir, 'metric_p%d.txt' % idx), 'w') as f:
            for l_ in p_loss_list:
                f.write(str(l_) + '\n')
        with open(os.path.join(result_dir, 'metric_m%d.txt' % idx), 'w') as f:
            for l_ in m_loss_list:
                f.write(str(l_) + '\n')
        with open(os.path.join(result_dir, 'metric_d%d.txt' % idx), 'w') as f:
            for l_ in dl_list:
                f.write(str(l_) + '\n')
        idx += 1

    l_mean = np.mean(metrics_l)
    p_mean = np.mean(metrics_p)
    m_mean = np.mean(metrics_m)
    d_mean = np.mean(metrics_d)
    with open(os.path.join(result_dir, 'metric_lmpd.txt'), 'w') as f:
        for i in range(len(metrics_l)):
            f.write(str(metrics_l[i])+'    '+str(metrics_m[i])+'    '+str(metrics_p[i])+'    '+str(metrics_d[i])+'\n')

    print('Overall metrics: loss_mean %f, ppl_mean %f, mse_mean %f, d_mean %f' % (l_mean, p_mean, m_mean, d_mean))
    with open(os.path.join(result_dir, 'mean_metrics.txt'), 'w') as f:
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
