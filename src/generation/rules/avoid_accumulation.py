import numpy as np

def avoid_accumulation(unresolved_note_ons, note_event, pitch, max_simultaneous_notes, best_event_time):
    note_off_event = None
    if note_event == 1 and len(unresolved_note_ons) >= max_simultaneous_notes:
        note_to_turn_off = unresolved_note_ons.pop()  # Remove and get an unresolved note
        note_off_event = [0, note_to_turn_off, 0, best_event_time]  # Create note-off event for it

    if note_event == 1:
        unresolved_note_ons.add(pitch)  # Add new note-on to unresolved
    elif note_event == 0 and pitch in unresolved_note_ons:
        unresolved_note_ons.remove(pitch)  # Remove resolved note-off

    return unresolved_note_ons, note_off_event
