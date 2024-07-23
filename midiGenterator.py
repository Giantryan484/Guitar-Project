import random
from music21 import stream, note, clef, midi, instrument

# Constants
LOWEST_NOTE = 40  # MIDI number for E2
HIGHEST_NOTE = 88  # MIDI number for E6
NOTE_PROBABILITY = 0.7  # Probability of a note being played instead of a rest
NUM_MEASURES = 4
NOTES_PER_MEASURE = 16  # 16 sixteenth-notes per measure
DURATIONS = [0.25, 0.5, 1.0, 2.0, 4.0]  # 16th, 8th, quarter, half, whole
MAX_ACTIVE_NOTES = 5

def biased_note_choice(): # Randomly select note, but certain note ranges are more favored
    ranges = {'low': (40, 60), 'middle': (61, 73), 'high': (74, 88)}
    probabilities = {'low': 0.5, 'middle': 0.35, 'high': 0.15}
    selected_range = random.choices(list(ranges.keys()), weights=list(probabilities.values()), k=1)[0]
    return random.randint(*ranges[selected_range]) #return random note range

def generate_melody():
    melody = stream.Part()
    melody.append(clef.TrebleClef())
    
    # Guitar intrument lines up with Bank 0 preset 24 (IDK why, but this is the only way I could get it working)
    gen_inst = instrument.Guitar()
    melody.insert(0, gen_inst)

    current_time = 0.0
    max_time = NUM_MEASURES * 4.0 # time is measured in quarter note proportions
    
    active_notes = [] # lets us keep track of num notes active

    while current_time < max_time:
        if random.random() < NOTE_PROBABILITY: # If note (else rest)
            note_duration = random.choice(DURATIONS)
            if current_time + note_duration > max_time:
                note_duration = max_time - current_time

            midi_note = biased_note_choice() # returns int note id
            new_note = note.Note(midi=midi_note, quarterLength=note_duration) 
            active_notes = [(start, end, n) for start, end, n in active_notes if end > current_time]
            
            if len(active_notes) < MAX_ACTIVE_NOTES: # If we can insert note (else rest)
                melody.insert(current_time, new_note)
                active_notes.append((current_time, current_time + note_duration, new_note))
            else:
                melody.insert(current_time, note.Rest(quarterLength=0.25))
        else:
            melody.insert(current_time, note.Rest(quarterLength=0.25))
        current_time += 0.25 # increment one 16th

    return melody

def generate_random_basic(filename):
    melody = generate_melody()

    # save to file
    midi_filename = 'MIDIs/' + filename + '.mid'
    mf = midi.translate.music21ObjectToMidiFile(melody)
    mf.open(midi_filename, 'wb')
    mf.write()
    mf.close()














# LEGACY CODE (just for my reference for when I inevitably screw something up):

# import random
# from music21 import stream, note, clef, midi, instrument
# # from mido import MidiFile, MetaMessage, Message

# # Constants
# LOWEST_NOTE = 40  # MIDI number for E2
# HIGHEST_NOTE = 88  # MIDI number for E6
# NOTE_PROBABILITY = 0.7  # Probability of a note being played instead of a rest
# NUM_MEASURES = 4
# NOTES_PER_MEASURE = 16  # 16 sixteenth-notes per measure
# DURATIONS = [0.25, 0.5, 1.0, 2.0, 4.0]  # 16th, 8th, quarter, half, whole
# MAX_ACTIVE_NOTES = 5

# def biased_note_choice():
#     ranges = {
#         'low': (40, 60),
#         'middle': (61, 73),
#         'high': (74, 88)
#     }
#     probabilities = {
#         'low': 0.5,
#         'middle': 0.35,
#         'high': 0.15
#     }
#     selected_range = random.choices(
#         population=list(ranges.keys()), 
#         weights=list(probabilities.values()), 
#         k=1
#     )[0]
    
#     return random.randint(*ranges[selected_range])

# def generate_melody():
#     melody = stream.Part()
#     melody.append(clef.TrebleClef())
    
#     # Create a generic Instrument, since music21 might not handle custom bank/program directly
#     gen_inst = instrument.Guitar()
#     # gen_inst.midiProgram = 26  # Acoustic guitar, adjust if needed
#     melody.insert(0, gen_inst)
    
#     # Manually insert a MIDI program change
#     # Bank select MSB (0) and LSB (32) followed by Program Change
#     # For General MIDI, bank numbers start from 0, multiplied by 128 for MSB
#     # bank_msb = midi.MidiControlChange(channel=0, control=0, value=7)
#     # bank_lsb = midi.MidiControlChange(channel=0, control=32, value=0)
#     # prog_change = midi.MidiProgramChange(channel=0, program=26)  # program number is 0-indexed, 27th program

#     # melody.insert(0, bank_msb)
#     # melody.insert(0, bank_lsb)
#     # melody.insert(0, prog_change)

#     current_time = 0.0
#     max_time = NUM_MEASURES * 4.0
    
#     active_notes = []

#     while current_time < max_time:
#         if random.random() < NOTE_PROBABILITY:
#             note_duration = random.choice(DURATIONS)
#             if current_time + note_duration > max_time:
#                 note_duration = max_time - current_time

#             midi_note = biased_note_choice()
#             new_note = note.Note(midi=midi_note, quarterLength=note_duration)
#             active_notes = [(start, end, n) for start, end, n in active_notes if end > current_time]
            
#             if len(active_notes) < MAX_ACTIVE_NOTES:
#                 melody.insert(current_time, new_note)
#                 active_notes.append((current_time, current_time + note_duration, new_note))
#             else:
#                 melody.insert(current_time, note.Rest(quarterLength=0.25))
#         else:
#             melody.insert(current_time, note.Rest(quarterLength=0.25))
#         current_time += 0.25

#     return melody

# # def modify_midi(file_path):
# #     # Load the MIDI file
# #     midi = MidiFile(file_path)
    
# #     # Access the first track in the MIDI file
# #     # print(midi.tracks)
# #     track = midi.tracks[1]
    
# #     # Adding track name and instrument change at the beginning of the track
# #     track.insert(3, MetaMessage('track_name', name='Guitar Track', time=0))
# #     track.insert(4, Message('program_change', program=26, time=0))  # Guitar program
    
# #     # Add bank select messages
# #     track.insert(5, Message('control_change', control=0, value=7, time=0))  # MSB for bank 7
# #     track.insert(6, Message('control_change', control=32, value=0, time=0))  # LSB for bank 0

# #     # Save the modified MIDI file
# #     midi.save(file_path)
# #     print(midi.tracks)



# def generate_random_basic(filename):
#     # Generate and save MIDI file
#     melody = generate_melody()
#     midi_filename = 'MIDIs/' + filename + '.mid'
#     # melody.show()

#     # Save as MIDI with custom bank and preset
#     mf = midi.translate.music21ObjectToMidiFile(melody)
#     mf.open(midi_filename, 'wb')
#     mf.write()
#     mf.close()

    # modify_midi(midi_filename)


# import random
# from mido import MidiFile, MidiTrack, Message, MetaMessage

# # Constants
# LOWEST_NOTE = 40  # MIDI number for E2
# HIGHEST_NOTE = 88  # MIDI number for E6
# NOTE_PROBABILITY = 0.7  # Probability of a note being played instead of a rest
# NUM_MEASURES = 4
# NOTES_PER_MEASURE = 16  # 16 sixteenth-notes per measure
# DURATIONS = [0.25, 0.5, 1.0, 2.0, 4.0]  # durations in terms of beats
# MAX_ACTIVE_NOTES = 5

# def biased_note_choice():
#     ranges = {'low': (40, 60), 'middle': (61, 73), 'high': (74, 88)}
#     probabilities = {'low': 0.5, 'middle': 0.35, 'high': 0.15}
#     selected_range = random.choices(list(ranges.keys()), weights=list(probabilities.values()), k=1)[0]
#     return random.randint(*ranges[selected_range]) #return random note range

# def generate_melody_mido():
#     mid = MidiFile()
#     track = MidiTrack()
#     mid.tracks.append(track)
    
#     # Add track name and instrument change
#     track.append(MetaMessage('track_name', name='Guitar Track', time=0))
#     track.append(Message('program_change', program=26, time=0))  # Guitar program (adjust if necessary)
    
#     # Add bank select messages (Control 0 for MSB, Control 32 for LSB)
#     track.append(Message('control_change', control=0, value=7, time=0))  # MSB for bank 7
#     track.append(Message('control_change', control=32, value=0, time=0))  # LSB for bank 0

#     ticks_per_beat = mid.ticks_per_beat  # Default 480 ticks per beat
#     ticks_per_16th = ticks_per_beat // 4  # Ticks per 16th note
#     max_ticks = NUM_MEASURES * 4 * ticks_per_beat  # Total ticks for 4 measures
    
#     current_time = 0  # Track the absolute time in ticks
#     active_notes = []

#     while current_time < max_ticks:
#         if random.random() < NOTE_PROBABILITY:
#             note_duration = int(random.choice(DURATIONS) * ticks_per_beat)
#             if current_time + note_duration > max_ticks:
#                 note_duration = max_ticks - current_time

#             midi_note = biased_note_choice()
            
#             # Remove notes that have ended
#             active_notes = [(start, end, n) for start, end, n in active_notes if end > current_time]
            
#             if len(active_notes) < MAX_ACTIVE_NOTES:
#                 delta_time = current_time - (sum([dur for _, dur, _ in active_notes]) if active_notes else 0)
#                 track.append(Message('note_on', note=midi_note, velocity=64, time=delta_time))
#                 track.append(Message('note_off', note=midi_note, velocity=64, time=note_duration))
#                 active_notes.append((current_time, current_time + note_duration, midi_note))
#                 current_time += note_duration  # Increment current_time by the note's duration
#             else:
#                 track.append(Message('note_off', note=0, velocity=0, time=ticks_per_16th))
#                 current_time += ticks_per_16th
#         else:
#             track.append(Message('note_off', note=0, velocity=0, time=ticks_per_16th))
#             current_time += ticks_per_16th


#     return mid



# def generate_random_basic(filename):
#     # Generate and save MIDI file
#     midi_file = generate_melody_mido()
#     midi_filename = 'MIDIs/' + filename + '.mid'
#     midi_file.save(midi_filename)


# it dont work :(