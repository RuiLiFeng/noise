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


def scale_alpha_exp(alpha, t):
    alpha = alpha.astype(np.float32)
    return tf.pow(alpha, t) / (tf.pow(alpha, t) + tf.pow((1-alpha).astype(np.float32), t))


def embed(batch_size, resolution, imgs, network, iteration, result_dir, seed=6600):
    tf.reset_default_graph()
    G_args = dnnlib.EasyDict(func_name='training.networks_stylegan2_alpha.G_main')
    G_args.fmap_base = 8 << 10
    print('Loading networks from "%s"...' % network)
    tflib.init_tf()
    G = tflib.Network('G', num_channels=3, resolution=128, **G_args)
    _, _, Gs = pretrained_networks.load_networks(network)
    G.copy_vars_from(Gs)
    img_in = tf.placeholder(tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    lr = tf.get_variable('lr', dtype=tf.float32, initializer=tf.constant(0.005))
    opt_Ts = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8)
    noise_vars = [var for name, var in G.components.synthesis.vars.items() if name.startswith('noise')]
    alpha_vars = [var for name, var in G.components.synthesis.vars.items() if name.endswith('alpha')]
    alpha_evals = [alpha.eval() for alpha in alpha_vars]

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.randomize_noise = False
    G_syn = G.components.synthesis

    rnd = np.random.RandomState(seed)
    dlatent_avg = [var for name, var in G.vars.items() if name.startswith('dlatent_avg')][0].eval()
    dlatent_avg = np.expand_dims(np.expand_dims(dlatent_avg, 0), 1)
    dlatent_avg = dlatent_avg.repeat(12, 1)
    dlatent = tf.get_variable('dlatent', dtype=tf.float32, initializer=tf.constant(dlatent_avg),
                              trainable=True)
    Ts = [tf.get_variable('T%d'% i, dtype=tf.float32, initializer=tf.constant(1.0)) for i in range(len(alpha_evals))]
    alpha_pre = [scale_alpha_exp(alpha_eval, T) for alpha_eval, T in zip(alpha_evals, Ts)]
    synth_img = G_syn.get_output_for(dlatent, is_training=False, alpha_pre=alpha_pre, **G_kwargs)
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
        grads = tf.gradients(loss, [dlatent]+Ts)
        train_op1 = opt.apply_gradients(zip([grads[0]], [dlatent]))
        train_op2 = opt_Ts.apply_gradients(zip(grads[1:], Ts))
        train_op = tf.group(train_op1, train_op2)
        # train_op1 = opt.minimize(loss, var_list=[dlatent])
        # train_op2 = opt_Ts.minimize(loss, var_list=Ts)
    reset_opt = tf.variables_initializer(opt.variables()+opt_Ts.variables())
    reset_dl = tf.variables_initializer([dlatent]+Ts)

    tflib.init_uninitialized_vars()
    # rnd = np.random.RandomState(seed)
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    idx = 0
    metrics_l = []
    metrics_p = []
    metrics_m = []
    metrics_d = []
    ac_list = []
    for img in imgs:
        img = np.expand_dims(img, 0)
        loss_list = []
        p_loss_list = []
        m_loss_list = []
        dl_list = []
        si_list = []
        # tflib.set_vars({alpha: alpha_np for alpha, alpha_np in zip(alpha_vars, alpha_evals)})
        tflib.run([reset_opt, reset_dl])
        tflib.set_vars({lr: 0.005})
        for i in range(iteration):
            loss_, p_loss_, m_loss_, dl_, si_, ac_, _ = tflib.run([loss, pcep_loss, mse_loss, dlatent, synth_img, Ts, train_op],
                                                             {img_in: img})
            if i > 3500:
                tflib.set_vars({lr: 0.002})
            loss_list.append(loss_)
            p_loss_list.append(p_loss_)
            m_loss_list.append(m_loss_)
            dl_loss_ = np.sum(np.square(dl_-dlatent_avg))
            dl_list.append(dl_loss_)
            acm_ = np.mean(ac_)
            if i % 500 == 0:
                si_list.append(si_)
            if i % 100 == 0:
                print('idx %d, Loss %f, mse %f, ppl %f, dl %f, TsMean %f, step %d' % (idx, loss_, m_loss_, p_loss_, dl_loss_, acm_, i))
        # print('T optimization:')
        # for i in range(1000):
        #     loss_, p_loss_, m_loss_, dl_, si_, ac_, _ = tflib.run(
        #         [loss, pcep_loss, mse_loss, dlatent, synth_img, Ts, train_op2],
        #         {img_in: img})
        #     if i % 500 == 0:
        #         si_list.append(si_)
        #     if i % 100 == 0:
        #         acm_ = np.mean(ac_)
        #         print('idx %d, Loss %f, mse %f, ppl %f, dl %f, TsMean %f, step %d' % (idx, loss_, m_loss_, p_loss_, dl_loss_, acm_, i))
        print('TsMean: %f, loss: %f, ppl: %f, mse: %f, d: %f' % (acm_,
                                                               loss_list[-1],
                                                               p_loss_list[-1],
                                                               m_loss_list[-1],
                                                               dl_list[-1]))
        metrics_l.append(loss_list[-1])
        metrics_p.append(p_loss_list[-1])
        metrics_m.append(m_loss_list[-1])
        metrics_d.append(dl_list[-1])
        ac_list.append(ac_)
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
        f.write(str(alpha_evals)+'\n')
        for i in range(len(metrics_l)):
            f.write(str(ac_list[i])+'    '+str(metrics_l[i])+'    '+str(metrics_m[i])+'    '+str(metrics_p[i])+'    '+str(metrics_d[i])+'\n')

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
