import os
import time

import wandb

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from pu.misc.utils import z_generator, kld, save_model


class CGenPU(object):
    """ Implementation of the Conditional Generative Positive and Unlabeled framework (CGenPU). """

    def __init__(self, input_dim, latent_dim, models, lr=1e-4, beta_1=0.5, beta_2=0.999, aux_str=1, z_type="uniform",
                 callback_data_viz=None, seed_size=10, save_model_freq=100, mean=True, logger=None):
        """
        Construct CGenPU instance.

        :param input_dim: Input data dimensions.
        :param latent_dim: Latent vector dimensions.
        :param models: Instances of Keras models. Expected order: Discriminator, Classifier and Generator.
        :param lr: The learning rate.
        :param beta_1: The exponential decay rate for the 1st moment estimates.
        :param beta_2: The exponential decay rate for the 2nd moment estimates.
        :param aux_str: The auxiliary classifier's weight.
        :param z_type: The distribution to sample latent vector ("normal" or "uniform").
        :param callback_data_viz: The function for intermediate visualization of the training progress.
        :param seed_size: The number of examples for visualization of their evolution during the training.
        :param save_model_freq: The number of epochs to skip before saving model parameters.
        :param mean: The reduction operation for the batch loss aggregation (sum or mean).
        :param logger: The central dashboard to keep track of the hyperparameters, predictions, and more (see wandb.ai).
        """

        # General information for logging and debugging
        self.method = "CGenPU"
        # Extract model parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.latent_class_dim = 2
        self.learning_rate = lr if type(lr) == tuple or type(lr) == list else (lr, lr, lr)
        self.beta_1 = beta_1 if type(beta_1) == tuple or type(beta_1) == list else (beta_1, beta_1, beta_1)
        self.beta_2 = beta_2 if type(beta_2) == tuple or type(beta_2) == list else (beta_2, beta_2, beta_2)
        self.aux_str = aux_str
        self.z_type = z_type
        self.mean = mean
        # Extract models
        self.D, self.A, self.G = models
        # Define optimizers
        self.D_opt = tf.keras.optimizers.Adam(lr=self.learning_rate[0], beta_1=self.beta_1[0], beta_2=self.beta_2[0])
        self.A_opt = tf.keras.optimizers.Adam(lr=self.learning_rate[1], beta_1=self.beta_1[1], beta_2=self.beta_2[1])
        self.G_opt = tf.keras.optimizers.Adam(lr=self.learning_rate[2], beta_1=self.beta_1[2], beta_2=self.beta_2[2])
        # Select loss function
        self.loss_fun = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # Monitor learning progress with static input
        self.seed_size = seed_size
        seed = z_generator((self.seed_size, self.latent_dim), z_type=self.z_type)
        self.seed_pos = tf.concat([seed, np.random.multinomial(1, [1, 0], size=self.seed_size)], axis=1)
        self.seed_neg = tf.concat([seed, np.random.multinomial(1, [0, 1], size=self.seed_size)], axis=1)
        del seed
        # Save data visualizer callback
        self.callback_data_viz = callback_data_viz
        # Save model frequency
        self.save_freq = save_model_freq

        # Log project to WandB
        self.run = logger
        # Default settings for WandB
        if self.run is None:
            os.environ["WANDB_SILENT"] = "true"
            os.environ["WANDB_MODE"] = "dryrun"
            project_name = f"dryrun-{time.time()}"
            self.run = wandb.init(
                project=project_name,
                config={
                    "Data dim.": self.input_dim,
                    "Latent dim.": self.latent_dim,
                    "Latent class": self.latent_class_dim,
                    "Learning rate": self.learning_rate,
                    "Beta_1": self.beta_1,
                    "Beta_2": self.beta_2,
                    "Auxiliary strength": self.aux_str,
                    "Z-type": self.z_type
                },
            )
        else:
            self.run.config.update(
                {
                    "Data dim.": self.input_dim,
                    "Latent dim.": self.latent_dim,
                    "Latent class": self.latent_class_dim,
                    "Learning rate": self.learning_rate,
                    "Beta_1": self.beta_1,
                    "Beta_2": self.beta_2,
                    "Auxiliary strength": self.aux_str,
                    "Z-type": self.z_type
                }
            )

    def fit(self, xp, xu, epochs, batch_size=128, verbose=False):
        """
        Trains the framework for a fixed number of epochs (iterations on a dataset).

        :param xp: The dataset of labeled positive examples.
        :param xu: The dataset of unlabeled examples.
        :param epochs: The number of epochs to train the model.
        :param batch_size: The number of samples per gradient update.
        :param verbose: The verbosity mode.
        :return: None
        """
        # Save data parameters
        self.run.config.update({"Batch size": batch_size, "Epochs": epochs})
        # Convert numpy array to tensor slices
        xu_dataset = tf.data.Dataset.from_tensor_slices(xu).shuffle(xu.shape[0]).padded_batch(batch_size, drop_remainder=True)
        xu_iter = xu_dataset.repeat().as_numpy_iterator()
        xp_dataset = tf.data.Dataset.from_tensor_slices(xp).shuffle(xp.shape[0])
        xp_iter = xp_dataset.repeat().batch(batch_size).as_numpy_iterator()
        # Batch information
        batch_count = xu_dataset.cardinality().numpy()

        # Training loop
        for epoch in range(epochs):
            # Training history
            Ld, Lg, La = 0, 0, 0

            # Batch loop
            for _ in tqdm(range(batch_count), total=batch_count, ncols=100,
                          desc=f"{self.method} - Epoch {epoch + 1:03d}/{epochs:03d}: ", disable=(not verbose)):
                # Get next batch of data
                xu_data_batch, xp_data_batch = next(xu_iter), next(xp_iter)
                # Get next batch of synthetic data
                Np = tf.concat([z_generator((xp_data_batch.shape[0], self.latent_dim), z_type=self.z_type),
                                np.random.multinomial(1, [1, 0], size=xp_data_batch.shape[0])], axis=1)
                Nn = tf.concat([z_generator((xu_data_batch.shape[0], self.latent_dim), z_type=self.z_type),
                                np.random.multinomial(1, [0, 1], size=xu_data_batch.shape[0])], axis=1)
                # Generate synthetic examples
                gp_data_batch = self.G(Np, training=False)
                gn_data_batch = self.G(Nn, training=False)
                # Update model D
                Ld_, _ = self.train_step_D(xp_data_batch, xu_data_batch, gp_data_batch, gn_data_batch)
                # Update model A
                La_, _ = self.train_step_A(xp_data_batch, xu_data_batch, gp_data_batch, gn_data_batch)
                # Update model G
                (Lg_, Lga_), _ = self.train_step_G(xp_data_batch, xu_data_batch)
                # Update training history
                Ld += Ld_
                La += self.aux_str * La_
                Lg += (Lg_ + self.aux_str * Lga_)
            # Log learning process
            history = {"Loss_D": Ld / batch_count, "Loss_A": La / batch_count, "Loss_G": Lg / batch_count, "Epoch": epoch + 1}
            if self.callback_data_viz is not None:
                gp = self.G(self.seed_pos, training=False)
                gn = self.G(self.seed_neg, training=False)
                history["Data"] = wandb.Image(self.callback_data_viz(np.concatenate([gp, gn]),
                                              np.append(np.ones(gp.shape[0]), np.zeros(gn.shape[0])),
                                              labels=("Positive", "Negative")))
            self.run.log(history)
            if self.save_freq is not None and epoch % self.save_freq == 0:
                self.run.save(save_model(self.G, dir=self.run.dir, step=(epoch + 1) // self.save_freq, overwrite=True))
                self.run.save(save_model(self.A, dir=self.run.dir, step=(epoch + 1) // self.save_freq, overwrite=True))
                self.run.save(save_model(self.D, dir=self.run.dir, step=(epoch + 1) // self.save_freq, overwrite=True))
        # Save final model
        self.run.save(save_model(self.G, dir=self.run.dir, step=None, overwrite=True))
        self.run.save(save_model(self.A, dir=self.run.dir, step=None, overwrite=True))
        self.run.save(save_model(self.D, dir=self.run.dir, step=None, overwrite=True))
        # Stop logging
        self.run.finish()

    @tf.function
    def train_step_D(self, xp_data_batch, xu_data_batch, gp_data_batch, gn_data_batch):
        """
        A single training step (batch update) for the discriminator.

        :param xp_data_batch: The batch of labeled positive examples.
        :param xu_data_batch: The batch of unlabeled examples.
        :param gp_data_batch: The batch of generated positive examples.
        :param gn_data_batch: The batch of generated negative examples.
        :return: Returns loss and gradients for logging.
        """
        # Start learning phase
        with tf.GradientTape() as tape:
            # Label examples as real or fake
            xp_real_batch_d = self.D(xp_data_batch, training=True)
            xu_real_batch_d = self.D(xu_data_batch, training=True)
            xp_fake_batch_d = self.D(gp_data_batch, training=True)
            xn_fake_batch_d = self.D(gn_data_batch, training=True)
            # Discriminator loss
            Lr = self.loss_fun(tf.ones_like(xp_real_batch_d), xp_real_batch_d) + \
                 self.loss_fun(tf.ones_like(xu_real_batch_d), xu_real_batch_d)
            Lf = self.loss_fun(tf.zeros_like(xp_fake_batch_d), xp_fake_batch_d) + \
                 self.loss_fun(tf.zeros_like(xn_fake_batch_d), xn_fake_batch_d)
            # Combine
            Ld = 0.5 * Lr + 0.5 * Lf
        # Apply gradients
        grad_d = tape.gradient(Ld, self.D.trainable_variables)
        self.D_opt.apply_gradients(zip(grad_d, self.D.trainable_variables))
        return Ld, grad_d

    @tf.function
    def train_step_G(self, xp_data_batch, xu_data_batch):
        """
        A single training step (batch update) for the generator.

        :param xp_data_batch: The batch of labeled positive examples.
        :param xu_data_batch: The batch of unlabeled examples.
        :return: Returns loss and gradients for logging.
        """
        # Sample random noise
        Np = tf.concat([z_generator((xp_data_batch.shape[0], self.latent_dim), z_type=self.z_type),
                        np.random.multinomial(1, [1, 0], size=xp_data_batch.shape[0])], axis=1)
        Nn = tf.concat([z_generator((xu_data_batch.shape[0], self.latent_dim), z_type=self.z_type),
                        np.random.multinomial(1, [0, 1], size=xu_data_batch.shape[0])], axis=1)
        # Start learning phase
        with tf.GradientTape() as tape:
            # Generate examples
            gp_data_batch = self.G(Np, training=True)
            gn_data_batch = self.G(Nn, training=True)
            # Label examples as real or fake
            xp_fake_batch_d = self.D(gp_data_batch, training=False)
            xn_fake_batch_d = self.D(gn_data_batch, training=False)
            # Label as positive or negative
            xp_real_batch_a = self.A(xp_data_batch, training=False)
            xp_fake_batch_a = self.A(gp_data_batch, training=False)
            xn_fake_batch_a = self.A(gn_data_batch, training=False)
            # Generator loss
            Lg = self.loss_fun(tf.ones_like(xp_fake_batch_d), xp_fake_batch_d) +\
                 self.loss_fun(tf.ones_like(xn_fake_batch_d), xn_fake_batch_d)
            # Auxiliary
            La = kld(xp_real_batch_a, xp_fake_batch_a, reduce_mean=self.mean) + \
                 kld(xp_fake_batch_a, 1. - xn_fake_batch_a, reduce_mean=self.mean)
            # Combine
            L = Lg + self.aux_str * La
        # Apply gradients
        grad_g = tape.gradient(L, self.G.trainable_variables)
        self.G_opt.apply_gradients(zip(grad_g, self.G.trainable_variables))
        return (Lg, La), grad_g

    @tf.function
    def train_step_A(self, xp_data_batch, xu_data_batch, gp_data_batch, gn_data_batch):
        """
        A single training step (batch update) for the auxiliary classifer.

        :param xp_data_batch: The batch of labeled positive examples.
        :param xu_data_batch: The batch of unlabeled examples.
        :param gp_data_batch: The batch of generated positive examples.
        :param gn_data_batch: The batch of generated negative examples.
        :return: Returns loss and gradients for logging.
        """
        # Start learning phase
        with tf.GradientTape() as tape:
            # Label examples as real or fake
            xp_real_batch_a = self.A(xp_data_batch, training=True)
            xp_fake_batch_a = self.A(gp_data_batch, training=True)
            xn_fake_batch_a = self.A(gn_data_batch, training=True)
            # Auxiliary loss
            La = kld(tf.ones_like(xp_real_batch_a), xp_real_batch_a, reduce_mean=self.mean) + \
                 kld(xp_real_batch_a, xp_fake_batch_a, reduce_mean=self.mean) + \
                 kld(xp_fake_batch_a, 1. - xn_fake_batch_a, reduce_mean=self.mean)
            L = self.aux_str * La
        # Apply gradients
        grad_a = tape.gradient(L, self.A.trainable_variables)
        self.A_opt.apply_gradients(zip(grad_a, self.A.trainable_variables))
        return La, grad_a

    def generate(self, N=100):
        """
        Returns dataset of labeled positive and negative examples.

        :param N: The size of the dataset.
        :return: Labeled artificial dataset.
        """
        # Sample random noise
        Np = tf.concat([z_generator((N, self.latent_dim), z_type=self.z_type), np.random.multinomial(1, [1, 0], size=N//2)], axis=1)
        Nn = tf.concat([z_generator((N, self.latent_dim), z_type=self.z_type), np.random.multinomial(1, [0, 1], size=N//2)], axis=1)
        # Generate data
        gp = self.G(Np, training=False)
        gn = self.G(Nn, training=False)
        # Concatenate and prepare labels
        X = np.concatenate([gp, gn])
        y = np.append(np.ones(gp.shape[0]), np.zeros(gn.shape[0]))
        return X, y

    def predict(self, X):
        """
        Returns the predicted probabilities of data being positive.

        :param X: The test data.
        :return: Probability of being positive.
        """
        if X is None:
            return None
        return self.A(X, training=False)
