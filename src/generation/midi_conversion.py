from mido import MidiFile, MidiTrack, Message

def convert_to_midi(sequence, output_path):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    for note, pitch, velocity, time in sequence:
        pitch = int(pitch)
        velocity = int(velocity)
        time = int(time)

        if note == 1:
            track.append(Message('note_on', note=pitch, velocity=velocity, time=time))
        else:
            track.append(Message('note_off', note=pitch, velocity=0, time=time))

    mid.save(output_path)
