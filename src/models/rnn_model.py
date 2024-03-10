import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Embedding, LayerNormalization, Bidirectional
from tensorflow.keras.regularizers import l2

def build_multi_output_rnn_model(input_shape, num_units, dropout_rate, pitch_range, velocity_range, time_delta_range, use_embeddings=False, embedding_dim=None):
    print("Building more powerful RNN model...")
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Optional embedding layer for categorical input
    if use_embeddings and embedding_dim is not None:
        x = Embedding(input_dim=pitch_range, output_dim=embedding_dim, input_length=input_shape[0])(x)
        print("Added Embedding layer with output_dim:", embedding_dim)

    # First GRU layer
    x = Bidirectional(GRU(num_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=0.1, kernel_regularizer=l2(0.01)))(x)
    x = LayerNormalization()(x)
    print(f"Added Bidirectional GRU layer with {num_units} units, dropout: {dropout_rate}, recurrent dropout: 0.1, L2 regularization.")

    # Additional GRU layers
    for _ in range(2):  # Adding two more GRU layers
        x = Bidirectional(GRU(num_units, return_sequences=True, dropout=dropout_rate, recurrent_dropout=0.1, kernel_regularizer=l2(0.01)))(x)
        x = LayerNormalization()(x)
        print(f"Added Bidirectional GRU layer with {num_units} units, dropout: {dropout_rate}, recurrent dropout: 0.1, L2 regularization.")

    # Final GRU layer without return_sequences
    x = Bidirectional(GRU(num_units, return_sequences=False, dropout=dropout_rate, recurrent_dropout=0.1, kernel_regularizer=l2(0.01)))(x)
    x = LayerNormalization()(x)
    print(f"Added Final Bidirectional GRU layer with {num_units} units, dropout: {dropout_rate}, recurrent dropout: 0.1, L2 regularization.")

    # Output layers with L2 regularization
    pitch_output = Dense(pitch_range, activation='softmax', name='pitch_output', kernel_regularizer=l2(0.01))(x)
    # Assuming velocity is normalized between 0 and 1, using sigmoid activation
    velocity_output = Dense(1, activation='sigmoid', name='velocity_output', kernel_regularizer=l2(0.01))(x)
    time_delta_output = Dense(time_delta_range, activation='softmax', name='time_delta_output', kernel_regularizer=l2(0.01))(x)

    model = Model(inputs=inputs, outputs=[pitch_output, velocity_output, time_delta_output])
    print("Enhanced model built with pitch, velocity, and time delta outputs.")

    return model
