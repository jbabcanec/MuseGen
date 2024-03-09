from mido import MidiFile, MidiTrack, Message
import numpy as np

def sequence_to_midi(sequence, output_file, ticks_per_beat=480, default_note_duration=10):
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    last_accumulated_time_delta = 0  # Initialize with the accumulated_time_delta of the first event
    max_tick = 0  # Initialize max tick tracker

    for i, event in enumerate(sequence):
        pitch, normalized_velocity, accumulated_time_delta, _ = event

        # For the first event, delta_time is just its accumulated_time_delta converted to ticks
        # For subsequent events, delta_time is the difference in accumulated_time_delta converted to ticks
        if i == 0:
            delta_time = int(accumulated_time_delta * ticks_per_beat)
        else:
            delta_time = int((accumulated_time_delta - last_accumulated_time_delta) * ticks_per_beat)

        print(f"Event {i}: accumulated_time_delta = {accumulated_time_delta}, delta_time = {delta_time}")

        # Update last_accumulated_time_delta for the next iteration
        last_accumulated_time_delta = accumulated_time_delta

        # Update max_tick tracker
        max_tick += delta_time
        print(f"Event {i}: max_tick so far = {max_tick}")

        # Rescale velocity back to MIDI standards
        velocity = int(normalized_velocity * 127)
        pitch = max(0, min(int(pitch), 127))
        velocity = max(0, min(velocity, 127))

        # Add 'note_on' and 'note_off' messages with calculated delta_time
        track.append(Message('note_on', note=pitch, velocity=velocity, time=delta_time))
        track.append(Message('note_off', note=pitch, velocity=0, time=default_note_duration))

    mid.save(output_file)
    print(f"MIDI file saved to {output_file}")
    print(f"The largest tick in the generated MIDI sequence is: {max_tick}")
