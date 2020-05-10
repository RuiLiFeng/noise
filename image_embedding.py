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
    I = np.array(I, np.float32) / 255.0
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


def embed(batch_size, resolution, img, G, iteration, vgg, seed=6600):
    img_in = tf.constant(img)
    opt = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    dlatent = tf.get_variable('dlatent', dtype=tf.float32, initializer=tf.zeros([1, 12, 512]),
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
    synth_img = G_syn.get_output_for(dlatent, is_training=False, **G_kwargs)
    synth_img = (synth_img + 1.0) / 2.0
    with tf.variable_scope('mse_loss'):
        mse_loss = tf.reduce_mean(tf.square(img_in - synth_img))
    with tf.variable_scope('perceptual_loss'):
        vgg_in = tf.concat([img_in, synth_img], 0)
        _ = vgg(vgg_in)
        h1 = vgg.get_layer('block1_conv1').output
        h2 = vgg.get_layer('block1_conv2').output
        h3 = vgg.get_layer('block3_conv2').output
        h4 = vgg.get_layer('block4_conv2').output
        pcep_loss = tf.reduce_mean(tf.square(h1[0] - h1[1])) + tf.reduce_mean(tf.square(h2[0] - h2[1])) + \
                    tf.reduce_mean(tf.square(h3[0] - h3[1])) + tf.reduce_mean(tf.square(h4[0] - h4[1]))
    loss = mse_loss + pcep_loss
    with tf.control_dependencies([loss]):
        train_op = opt.minimize(loss, var_list=[dlatent])

    tflib.init_uninitialized_vars()
    rnd = np.random.RandomState(seed)
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
    for i in range(iteration):
        loss_, p_loss_, m_loss_, dl_, si_, _ = tflib.run([loss, pcep_loss, mse_loss, dlatent, synth_img, train_op])
        loss_list.append(loss_)
        p_loss_list.append(p_loss_)
        m_loss_list.append(m_loss_)
        dl_list.append(dl_)
        if i % 500 == 0:
            si_list.append(si_)
        if i % 100 == 0:
            print('Loss %f, mse %f, ppl %f, step %d' % (loss_, m_loss_, p_loss_, i))
    return loss_list, p_loss_list, m_loss_list, dl_list, si_list


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual loss')
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--resolution', default=128, type=int)
    parser.add_argument('--src_dir', default="/gdata2/fengrl/imgs-for-embed/")
    parser.add_argument('--network', default="weight_files/pytorch/karras2019stylegan-ffhq-1024x1024.pt", type=str)
    parser.add_argument('--iteration', default=1000, type=int)
    parser.add_argument('--result_dir', default='/gdata2/fengrl/inverse')

    args = parser.parse_args()

    print('Loading networks from "%s"...' % args.network)
    tflib.init_tf()
    _, _, G = pretrained_networks.load_networks(args.network)
    # vgg = misc.load_pkl('/gdata2/fengrl/metrics/vgg16_zhang_perceptual.pkl')
    tf.keras.backend.set_image_data_format('channels_first')
    vgg = tf.keras.applications.VGG16(include_top=False, #input_shape=(3, 128, 128),
                                      weights='/gdata2/fengrl/metrics/vgg.h5',
                                      pooling=None)

    imgs = read_images(args.src_dir)

    metrics = []

    for img in imgs:
        img = np.expand_dims(img, 0)
        l, p, m, d, s = embed(args.batch_size, args.resolution, img, G, args.iteration, vgg)
        misc.save_image_grid(np.concatenate(s, 0), os.path.join(args.result_dir, 'si.png'), drange=[0, 1])
        break


if __name__ == "__main__":
    main()
