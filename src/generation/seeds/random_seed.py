import numpy as np
import random

def generate_random_seed(sequence_length=100, most_likely_first_pitch=60, pitch_range=128, time_increment=0.1):
    current_time = 0  # Initialize the current time
    seed_sequence = []

    # Start with an initial event with the most likely pitch, full velocity, and initial time
    seed_sequence.append([1, most_likely_first_pitch, 1.0, current_time])

    # Generate subsequent events with incremented time values
    for _ in range(1, sequence_length):
        pitch = np.random.randint(0, pitch_range)  # Random pitch within the range
        velocity = np.random.random()  # Random velocity between 0 and 1
        current_time += time_increment  # Increment the current time

        # Generate a new event with a note-on, random pitch, random velocity, and updated current time
        event = [1, pitch, velocity, current_time]
        seed_sequence.append(event)

    return np.array(seed_sequence)