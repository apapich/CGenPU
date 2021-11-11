import os
import wandb

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons

from pu.CGenPU import CGenPU
from pu.misc import utils


def load_moons(N):
    x, y = make_moons((N, N), noise=0.1)
    xp = x[y == 1, :]
    xu = x[y != 1, :]

    indices = np.random.choice(range(0, xp.shape[0]), size=int(xp.shape[0] * 0.7), replace=False)
    xu = np.concatenate([xu, xp[indices]])
    xp = np.delete(xp, indices, axis=0)

    xu, xp = xu.astype("float32"), xp.astype("float32")
    return xp, xu


def build_discriminator(input_dim):
    xi = tf.keras.layers.Input(shape=input_dim)
    nn = tf.keras.layers.Dense(128)(xi)
    nn = tf.keras.layers.ReLU()(nn)
    nn = tf.keras.layers.Dense(128)(nn)
    nn = tf.keras.layers.ReLU()(nn)
    nn = tf.keras.layers.Dense(128)(nn)
    nn = tf.keras.layers.ReLU()(nn)
    nn = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(nn)
    return tf.keras.models.Model(xi, nn, name="D")


def build_generator(input_dim, latent_dim):
    zi = tf.keras.layers.Input(shape=latent_dim)
    nn = tf.keras.layers.Dense(128)(zi)
    nn = tf.keras.layers.ReLU()(nn)
    nn = tf.keras.layers.Dense(128)(nn)
    nn = tf.keras.layers.ReLU()(nn)
    nn = tf.keras.layers.Dense(128)(nn)
    nn = tf.keras.layers.ReLU()(nn)
    nn = tf.keras.layers.Dense(input_dim, activation="linear")(nn)
    return tf.keras.models.Model(zi, nn, name="G")


def build_classifier(input_dim):
    xi = tf.keras.layers.Input(shape=input_dim)
    nn = tf.keras.layers.Dense(128)(xi)
    nn = tf.keras.layers.ReLU()(nn)
    nn = tf.keras.layers.Dense(128)(nn)
    nn = tf.keras.layers.ReLU()(nn)
    nn = tf.keras.layers.Dense(128)(nn)
    nn = tf.keras.layers.ReLU()(nn)
    nn = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(nn)
    return tf.keras.models.Model(xi, nn, name="A")


def plot_points(X, y, alpha=.3, labels=("Unlabeled", "Positive")):
    if tf.is_tensor(X):
        X = np.array(X)
    fig = plt.figure(figsize=(10, 10))
    plt.plot(X[y != 1, 0], X[y != 1, 1], "o", label=labels[0], alpha=alpha)
    plt.plot(X[y == 1, 0], X[y == 1, 1], "ro", label=labels[1], alpha=alpha)
    plt.xlabel("")
    plt.ylabel("")
    plt.legend()
    x1, x2 = tf.reduce_min(X[:, 0]), tf.reduce_max(X[:, 0])
    y1, y2 = tf.reduce_min(X[:, 1]), tf.reduce_max(X[:, 1])
    plt.xlim((x1, x2))
    plt.ylim((y1, y2))

    return utils.plot_to_image(fig)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["WANDB_SILENT"] = "true"

    run = wandb.init(
        project="CGenPU",
        group="moons"
    )

    callback = lambda X, y, labels=None: plot_points(X, y, labels=labels)

    xp, xu = load_moons(500)

    D = build_discriminator(2)
    G = build_generator(2, 8 + 2)
    A = build_classifier(2)

    g = CGenPU(2, 8, (D, A, G), lr=1e-4, aux_str=1, callback_data_viz=callback, seed_size=500, logger=run, mean=True)

    try:
        g.fit(xp, xu, epochs=1000, batch_size=128, verbose=True)
    except KeyboardInterrupt:
        pass
    finally:
        del g