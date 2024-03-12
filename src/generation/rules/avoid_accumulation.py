import numpy as np

def avoid_accumulation(sequence, max_simultaneous_notes):
    # Track the state and count of "note on" events
    note_on_count = 0
    note_states = {}

    corrected_sequence = []
    for event in sequence:
        note_event, pitch = int(event[0]), int(event[1])

        # Check for "note on" event
        if note_event == 1:
            # If adding this "note on" event exceeds the limit
            if note_on_count >= max_simultaneous_notes:
                # Find a "note on" event to turn off before adding a new one
                for resolved_pitch in note_states.keys():
                    if note_states[resolved_pitch]:  # If the note is currently on
                        # Add a "note off" event for the resolved note
                        corrected_sequence.append([0, resolved_pitch, 0, event[3]])  # Use the current event's time for "note off"
                        note_states[resolved_pitch] = False  # Mark the note as off
                        note_on_count -= 1  # Decrement the "note on" count
                        break  # Break after resolving one note to add the new "note on" event

            # Add the "note on" event
            note_states[pitch] = True
            note_on_count += 1
            corrected_sequence.append(event)

        # Check for "note off" event
        elif note_event == 0:
            # If the note is currently on, turn it off
            if pitch in note_states and note_states[pitch]:
                note_states[pitch] = False
                note_on_count -= 1
                corrected_sequence.append(event)

    return np.array(corrected_sequence)
