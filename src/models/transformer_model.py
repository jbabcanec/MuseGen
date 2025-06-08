import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                     MultiHeadAttention, Add, Flatten)
from tensorflow.keras.models import Model


def positional_encoding(length, depth):
    depth = depth // 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


def transformer_block(x, num_heads, ff_dim, dropout_rate):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn_output = Dense(ff_dim * 2, activation='relu')(x)
    ffn_output = Dense(ff_dim)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    x = Add()([x, ffn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def build_transformer_model(
    input_shape,
    num_heads=4,
    ff_dim=128,
    num_layers=4,
    dropout_rate=0.1,
    pitch_range=128,
    velocity_range=128,
    time_delta_range=3500,
):
    """Build a simple Transformer-based model for music generation."""
    seq_len, num_features = input_shape
    inputs = Input(shape=input_shape)

    x = Dense(ff_dim)(inputs)
    x += positional_encoding(seq_len, ff_dim)

    for _ in range(num_layers):
        x = transformer_block(x, num_heads, ff_dim, dropout_rate)

    x = Flatten()(x)

    note_event_output = Dense(2, activation='softmax', name='note_event_output')(x)
    pitch_output = Dense(pitch_range, activation='softmax', name='pitch_output')(x)
    velocity_output = Dense(velocity_range, activation='softmax', name='velocity_output')(x)
    time_delta_output = Dense(time_delta_range, activation='softmax', name='time_delta_output')(x)

    model = Model(inputs=inputs, outputs=[note_event_output, pitch_output, velocity_output, time_delta_output])
    return model
