import os
import warnings

import cv2
import imageio
import numpy as np
from scipy.stats import truncnorm

from config import MP4_CODEC

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    import tensorflow_hub as hub

# 0 - INFO, 1 - WARNING, 2 - ERROR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BigGAN:
    def __init__(self, data_path, biggan_size=256):
        self.biggan_size = biggan_size
        tf.reset_default_graph()
        # saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=3, reshape=False)
        module_url = F'https://tfhub.dev/deepmind/biggan-{biggan_size}/2'
        module_path = os.path.join(data_path, f'biggan_{biggan_size}')

        if not os.path.exists(module_path):
            # print('Loading BigGAN module from:', module_url)
            module = hub.Module(module_url)
        else:
            # print("Loading BigGAN module from local storage.")
            module = hub.Module(module_path)

        inputs = {k: tf.compat.v1.placeholder(v.dtype, v.get_shape().as_list(), k)
                for k, v in module.get_input_info_dict().items()}
        self.output = module(inputs)

        self.input_z = inputs['z']
        self.input_y = inputs['y']
        self.input_trunc = inputs['truncation']

        self.dim_z = self.input_z.shape.as_list()[1]
        self.vocab_size = self.input_y.shape.as_list()[1]
        self.graph = tf.get_default_graph()

        if not os.path.exists(module_path):
            print(F"Saving {module_path} locally.")
            with tf.compat.v1.Session(graph=self.graph) as sess:
                sess.run(tf.global_variables_initializer())
                module.export(F'biggan_{biggan_size}', sess)

    def truncated_z_sample(self, batch_size, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size, self.dim_z), random_state=state)
        return truncation * values

    def one_hot(self, index, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        index = np.asarray(index)
        if len(index.shape) == 0:
            index = np.asarray([index])
        assert len(index.shape) == 1
        num = index.shape[0]
        output = np.zeros((num, vocab_size), dtype=np.float32)
        output[np.arange(num), index] = 1
        return output

    def one_hot_if_needed(self, label, vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        label = np.asarray(label)
        if len(label.shape) <= 1:
            label = self.one_hot(label, vocab_size)
        assert len(label.shape) == 2
        return label

    def sample(self, sess, noise, label, truncation=1., batch_size=8,
               vocab_size=None):
        if vocab_size is None:
            vocab_size = self.vocab_size
        noise = np.asarray(noise)
        label = np.asarray(label)
        num = noise.shape[0]
        if len(label.shape) == 0:
            label = np.asarray([label] * num)
        if label.shape[0] != num:
            raise ValueError('Got # noise samples ({}) != # label samples ({})'
        .format(noise.shape[0], label.shape[0]))
        label = self.one_hot_if_needed(label, vocab_size)
        ims = []
        for batch_start in range(0, num, batch_size):
            s = slice(batch_start, min(num, batch_start + batch_size))
            feed_dict = {self.input_z: noise[s], self.input_y: label[s], self.input_trunc: truncation}
            ims.append(sess.run(self.output, feed_dict=feed_dict))
        ims = np.concatenate(ims, axis=0)
        assert ims.shape[0] == num
        ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
        ims = np.uint8(ims)
        return ims

    def interpolate(self, A, B, num_interps):
        alphas = np.linspace(0, 1, num_interps)
        if A.shape != B.shape:
            raise ValueError('A and B must have the same shape to interpolate.')
        return np.array([(1-a)*A + a*B for a in alphas])

    def interpolate_and_shape(self, A, B, num_interps, num_samples):
        interps = self.interpolate(A, B, num_interps)
        return (interps.transpose(1, 0, *range(2, len(interps.shape)))
                        .reshape(num_samples * num_interps, *interps.shape[2:]))

    def gif_img_array(self, filename, ims, fps=12):
        frame_arr = []
        frames = ims.shape[0]
        for i in range(0, frames):
            tmp_img = np.asarray(ims[i, ], dtype=np.uint8)
            if (i == 0) or (i == frames-1):
                for _ in range(int(fps/1.5)):
                    frame_arr.append(tmp_img)
            frame_arr.append(tmp_img)
        imageio.mimsave(filename, frame_arr, 'GIF', fps=fps)

    def mp4_img_array(self, filename, ims, fps=12):
        width = height = self.biggan_size
        frames = ims.shape[0]
        cc = cv2.VideoWriter_fourcc(*MP4_CODEC)
        out = cv2.VideoWriter(filename, cc, float(fps), (width, height))
        if not os.path.exists(filename):
            out.release()
            raise Exception('Cannot create video, codec is likely missing, try using GIF instead.')
        for i in range(0, frames):
            a = ims[i, ]
            tmp_img = cv2.cvtColor(np.asarray(a, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            # Add extra frames
            if (i == 0) or (i == frames-1):
                for _ in range(int(fps/1.5)):
                    out.write(tmp_img)
            out.write(tmp_img)
        out.release()

    def hell(self, filename, video=False, cat_a=206, samples=10, truncation=0.2, noise_a=3, fps=10):
        z = self.truncated_z_sample(samples, truncation, noise_a)
        with tf.compat.v1.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            ims = self.sample(sess, z, cat_a, truncation=truncation)

        if video:
            self.mp4_img_array(filename, ims, fps)
        else:
            self.gif_img_array(filename, ims, fps)

    def transform(self, filename, video=False, cat_a=206, cat_b=8, samples=1, interps=10, truncation=0.2, noise_a=0, noise_b=0, fps=10):
        z_A, z_B = [self.truncated_z_sample(samples, truncation, noise_seed)
                    for noise_seed in [noise_a, noise_b]]
        y_A, y_B = [self.one_hot([category] * samples)
                    for category in [cat_a, cat_b]]

        z_interp = self.interpolate_and_shape(z_A, z_B, interps, samples)
        y_interp = self.interpolate_and_shape(y_A, y_B, interps, samples)
        with tf.compat.v1.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            ims = self.sample(sess, z_interp, y_interp, truncation=truncation)

        if video:
            self.mp4_img_array(filename, ims, fps)
        else:
            self.gif_img_array(filename, ims, fps)

    def slerp(self, A, B, num_interps):
        alphas = np.linspace(-1.5, 2.5, num_interps)
        omega = np.zeros((A.shape[0], 1))
        for i in range(A.shape[0]):
            tmp = np.dot(A[i], B[i])/(np.linalg.norm(A[i])*np.linalg.norm(B[i]))
            omega[i] = np.arccos(np.clip(tmp, 0.0, 1.0))+1e-9
        return np.array([(np.sin((1-a)*omega)/np.sin(omega))*A + (np.sin(a*omega)/np.sin(omega))*B for a in alphas])

    def slerp_and_shape(self, A, B, num_interps):
        interps = self.slerp(A, B, num_interps)
        return (interps.transpose(1, 0, *range(2, len(interps.shape)))
                       .reshape(num_interps, *interps.shape[2:]))

    def generate_slerp(self, filename, video=False, cat_a=206, samples=1, interps=40, truncation=0.2, noise_a=3, noise_b=31, fps=10):
        z_A, z_B = [self.truncated_z_sample(samples, truncation, noise_seed)
                    for noise_seed in [noise_a, noise_b]]
        z_slerp = self.slerp_and_shape(z_A, z_B, interps)
        with tf.compat.v1.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            ims = self.sample(sess, z_slerp, self.one_hot([cat_a] * interps), truncation=truncation)

        if video:
            self.mp4_img_array(filename, ims, fps)
        else:
            self.gif_img_array(filename, ims, fps)
