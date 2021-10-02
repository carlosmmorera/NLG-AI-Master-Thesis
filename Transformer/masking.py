import tensorflow as tf


def create_padding_mask(seq):
    # tf.math.equal(seq, 0) genera una matriz cuyo valor es True en aquellas posiciones en que en la matriz seq
    # sea 0
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    # Produce una matriz triangular superior con la diagonal nula
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
