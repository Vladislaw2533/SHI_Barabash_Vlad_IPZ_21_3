# -*- coding: utf-8 -*-
"""LR_8_Task_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hy3CXYRgE8KUMy20PJ1D2l31UcIRE5j9
"""

import numpy as np
import tensorflow as tf

rng = np.random.default_rng(12345)
tf.random.set_seed(12345)

features = rng.random((1000, 1), dtype=np.float32)
targets = 2.0 * features + 1.0 + rng.normal(0, 0.1, (1000, 1))

weight = tf.Variable(tf.random.uniform([1], -1.0, 1.0), name="weight")
bias = tf.Variable(tf.zeros([1]), name="bias")

def model(inputs):
    return weight * inputs + bias

sgd_optimizer = tf.optimizers.SGD(learning_rate=0.05)

for iteration in range(20000):
    with tf.GradientTape() as gradient_tape:
        predicted = model(features)
        mse_loss = tf.reduce_mean(tf.square(targets - predicted))

    grads = gradient_tape.gradient(mse_loss, [weight, bias])
    sgd_optimizer.apply_gradients(zip(grads, [weight, bias]))

    if iteration % 1000 == 0:
        print(
            f"Iteration {iteration}: Loss={mse_loss.numpy():.4f}, Weight={weight.numpy()[0]:.4f}, Bias={bias.numpy()[0]:.4f}"
        )

print(f"\nTrained Parameters: Weight={weight.numpy()[0]:.4f}, Bias={bias.numpy()[0]:.4f}")