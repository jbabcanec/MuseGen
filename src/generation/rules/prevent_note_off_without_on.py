import numpy as np

def prevent_note_off_without_on(sequence):
    # Track the state of each note (on or off)
    note_states = {}

    corrected_sequence = []
    for event in sequence:
        note_event, pitch = int(event[0]), int(event[1])

        if note_event == 1:  # Note-on event
            # Mark the note as on
            note_states[pitch] = True
            corrected_sequence.append(event)
        elif note_event == 0:  # Note-off event
            # Check if the note is currently on before turning it off
            if pitch in note_states and note_states[pitch]:
                note_states[pitch] = False  # Mark the note as off
                corrected_sequence.append(event)
            # If the note was not on, ignore this note-off event

    return np.array(corrected_sequence)
