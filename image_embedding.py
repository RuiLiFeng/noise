import tensorflow as tf
import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
from training import misc

import re
import sys

import pretrained_networks


def read_image(img):
    I = PIL.Image.open(img)
    I = I.convert('RGB')
    I = np.array(I, np.float32) / 255.0
    if I.shape[-1] == 3:
        I.transpose([2, 0, 1])
    return I


def read_images(src_dir):
    return src_dir


def embed(batch_size, resolution, img, G, iteration, vgg, seed=6600):
    img_in = tf.constant(img)
    opt = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    dlatent = tf.get_variable('dlatent', shape=[1, 18, 512], dtype=tf.float32, initializer=tf.zeros([1, 18, 512]),
                              trainable=True)
    noise_vars = [var for name, var in G.components.synthesis.vars.items() if name.startswith('noise')]

    G_kwargs = dnnlib.EasyDict()
    G_kwargs.randomize_noise = False
    G_syn = G.components.synthesis
    loss_list = []
    p_loss_list = []
    m_loss_list = []
    dl_list = []
    si_list = []
    rnd = np.random.RandomState(seed)
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    synth_img = G_syn.get_output_for(dlatent, is_training=False, **G_kwargs)
    synth_img = (synth_img + 1.0) / 2.0
    with tf.variable_scope('mse_loss'):
        mse_loss = tf.reduce_mean(tf.square(img_in - synth_img))
    with tf.variable_scope('perceptual_loss'):
        pcep_loss = tf.reduce_sum(vgg.get_ouput_for(img_in, synth_img))
    loss = mse_loss + pcep_loss
    with tf.control_dependencies([loss]):
        train_op = opt.minimize(loss, var_list=[dlatent])

    tflib.init_uninitialized_vars()
    for i in range(iteration):
        loss_, p_loss_, m_loss_, dl_, si_, _ = tflib.run([loss, pcep_loss, mse_loss, dlatent, synth_img, train_op])
        loss_list.append(loss_)
        p_loss_list.append(p_loss_)
        m_loss_list.append(m_loss_)
        dl_list.append(dl_)
        si_list.append(si_)
        if i % 100 == 0:
            print('Loss %f, mse %f, ppl %f, step %d' % (loss_, m_loss_, p_loss_, i))
        misc.save_image_grid(np.concatenate(si_list, 0),
                             dnnlib.make_run_dir_path('si.png' % seed), drange=[0, 1])
    return loss_list, p_loss_list, m_loss_list, dl_list, si_list


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--resolution', default=128, type=int)
    parser.add_argument('--src_dir', default="source_image/")
    parser.add_argument('--network', default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt", type=str)
    parser.add_argument('--iteration', default=1000, type=int)

    args = parser.parse_args()

    print('Loading networks from "%s"...' % args.network)
    tflib.init_tf()
    _, _, G = pretrained_networks.load_networks(args.network)
    vgg = misc.load_pkl('/gdata2/fengrl/metrics/vgg16_zhang_perceptual.pkl')

    imgs = read_images(args.src_dir)

    metrics = []

    for img in imgs:
        l, p, m, d, s = embed(args.batch_size, args.resolution, img, G, args.iteration, vgg)


if __name__ == "__main__":
    main()