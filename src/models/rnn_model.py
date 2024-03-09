import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Embedding, BatchNormalization
from tensorflow.keras.regularizers import l2

def build_multi_output_rnn_model(input_shape, num_units, dropout_rate, pitch_range, velocity_range, time_delta_range, use_embeddings=False, embedding_dim=None):
    print("Building RNN model...")
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Optional embedding layer for categorical input
    if use_embeddings and embedding_dim is not None:
        x = Embedding(input_dim=pitch_range, output_dim=embedding_dim, input_length=input_shape[0])(x)
        print("Added Embedding layer with output_dim:", embedding_dim)

    # GRU layers with dropout and L2 regularization
    x = GRU(num_units, return_sequences=True, recurrent_dropout=0.1, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    print(f"Added GRU layer with {num_units} units, dropout: {dropout_rate}, recurrent dropout: 0.1, L2 regularization.")

    x = GRU(num_units, return_sequences=False, recurrent_dropout=0.1, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    print(f"Added GRU layer with {num_units} units, dropout: {dropout_rate}, recurrent dropout: 0.1, L2 regularization.")

    # Output layers with L2 regularization
    pitch_output = Dense(pitch_range, activation='softmax', name='pitch_output', kernel_regularizer=l2(0.01))(x)
    # Updated velocity output layer for continuous value prediction
    velocity_output = Dense(1, activation='sigmoid', name='velocity_output', kernel_regularizer=l2(0.01))(x)
    time_delta_output = Dense(time_delta_range, activation='softmax', name='time_delta_output', kernel_regularizer=l2(0.01))(x)

    model = Model(inputs=inputs, outputs=[pitch_output, velocity_output, time_delta_output])
    print("Model built with pitch, velocity, and time delta outputs.")

    return model