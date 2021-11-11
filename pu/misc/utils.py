import io
import tensorflow as tf
import matplotlib.pyplot as plt


def z_generator(shape, z_type="uniform"):
    """
    Samples from the selected distribution.

    :param shape: The dimensions of the sample.
    :param z_type: The distribution to sample from.
    :return: The random value matrix/vector.
    """
    if z_type == "uniform":
        return tf.random.uniform(shape=shape, minval=-1.0, maxval=1.0)
    if z_type == "normal":
        return tf.random.normal(shape=shape, mean=0.0, stddev=1.0)
    return None


def kld(P, Q, f=1e-7, reduce_mean=False):
    """
    Computes Kullback–Leibler divergence.

    :param P: The probability distribution.
    :param Q: The probability distribution.
    :param f: The offset value to avoid computational issues with log(0).
    :param reduce_mean: The aggregation type.
    :return: Computed KL divergence value.
    """
    P_ = tf.clip_by_value(P, clip_value_min=f, clip_value_max=1)
    Q_ = tf.clip_by_value(Q, clip_value_min=f, clip_value_max=1)

    if reduce_mean:
        return tf.reduce_mean(P_ * tf.math.log(P_ / Q_))
    return tf.reduce_sum(P_ * tf.math.log(P_ / Q_))


def save_model(model, dir="./tmp", step=0, overwrite=False):
    """
    Saves model parameters.

    :param model: The training model.
    :param dir: The destination.
    :param step: The optimization step.
    :param overwrite: Overwrite existing.
    :return: File path.
    """
    filepath = f"{dir}/{model.name}_{step}.h5" if step is not None else f"{dir}/{model.name}.h5"
    tf.keras.models.save_model(model, filepath=filepath, overwrite=overwrite)
    return filepath


def plot_to_image(figure):
    """
    Converts the matplotlib plot figure to a PNG image.

    :param figure: Matplotlib figure object.
    :return: Tensorflow image.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image
