def generate_music(models, seed_sequence, num_generate=100, temperature=1.0):
    input_sequence = np.array(seed_sequence)
    generated_sequence = []

    for i in range(num_generate):
        print(f"Generating event {i+1}/{num_generate}")

        for model in models:
            prediction = model.predict(np.expand_dims(input_sequence, axis=0))
            note_event_pred, pitch_pred, velocity_pred, event_time_pred = prediction

            # Apply softmax with temperature to note_event_pred probabilities
            note_event_prob = softmax(note_event_pred[0], temperature=temperature)
            pitch_prob = softmax(pitch_pred[0], temperature=temperature)

            # Sample a note event and pitch based on adjusted probabilities
            note_event = np.random.choice(range(len(note_event_prob)), p=note_event_prob)
            pitch = np.random.choice(range(len(pitch_prob)), p=pitch_prob)

            # For velocity and event time, you can still use the deterministic approach or adjust similarly
            best_velocity = velocity_pred[0][0] * 127  # Scaling factor for MIDI velocity
            best_event_time = event_time_pred[0].dot(np.arange(event_time_pred.shape[1]))  # Expected value for event time

            print(f"  Model: {model.name} - Note Event: {note_event}, Pitch: {pitch}, Velocity: {best_velocity:.2f}, Event Time: {best_event_time}")

        next_event = [note_event, pitch, best_velocity, best_event_time]
        generated_sequence.append(next_event)
        input_sequence = np.vstack([input_sequence[1:], next_event])

        print(f"Selected for event {i+1}: Note Event {note_event}, Pitch {pitch}, Velocity (scaled) {best_velocity:.2f}, Event Time {best_event_time}")
        print("------")

    return generated_sequence