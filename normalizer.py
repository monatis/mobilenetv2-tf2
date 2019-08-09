import tensorflow as tf
import numpy as np
import os

def get_mean_std(base_dir, filenames, target_size):
    n = 0
    r_mean, g_mean, b_mean = 0.0, 0.0, 0.0
    r_M2, g_M2, b_M2 = 0.0, 0.0, 0.0

    
    for z, filename in enumerate(filenames):
        if z % 1000 == 0:
            print("Processing image {}/{}".format(z+1, len(filenames)))

        x = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(os.path.join(base_dir, filename), target_size=target_size))
        r = x[:, :, 0].flatten().tolist()
        g = x[:, :, 1].flatten().tolist()
        b = x[:, :, 2].flatten().tolist()

        for (xr, xg, xb) in zip(r, g, b):
            n = n + 1

            r_delta = xr - r_mean
            g_delta = xg - g_mean
            b_delta = xb - b_mean

            r_mean = r_mean + r_delta/n
            g_mean = g_mean + g_delta/n
            b_mean = b_mean + b_delta/n

            r_M2 = r_M2 + r_delta * (xr - r_mean)
            g_M2 = g_M2 + g_delta * (xg - g_mean)
            b_M2 = b_M2 + b_delta * (xb - b_mean)

    r_variance = r_M2 / (n - 1)
    g_variance = g_M2 / (n - 1)
    b_variance = b_M2 / (n - 1)

    r_std = np.sqrt(r_variance)
    g_std = np.sqrt(g_variance)
    b_std = np.sqrt(b_variance)

    return np.array([r_mean, g_mean, b_mean]), np.array([r_std, g_std, b_std])


class Normalizer():
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if self.mean is not None:
            img = self.center(img)
        if self.std is not None:
            img = self.scale(img)
        return img

    def center(self, img):
        return img - self.mean

    def scale(self, img):
        return img / self.std

    def set_stats(self, mean, std):
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
        

    def get_stats(self, base_dir, filenames, target_size, calc_mean=True, calc_std=True):
        print("Calculating mean and standard deviation with shape: ", target_size)
        m, s = get_mean_std(base_dir, filenames, target_size)
        if calc_mean:
            self.mean = m
            self.mean = self.mean.reshape(1, 1, 3)
            print("Dataset mean [r, g, b] = {}".format(m.tolist()))
        if calc_std:
            self.std = s
            self.std = self.std.reshape(1, 1, 3)
            print("Dataset std [r, g, b] = {}". format(s.tolist()))

        return str(m.tolist()), str(s.tolist())
