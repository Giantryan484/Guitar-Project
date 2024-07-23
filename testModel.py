import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
from PIL import Image
import tensorflow as tf
import mido

# ---WAV PROCESSING---

def create_amplitude_tensors(filename, bpm):
    wav_file = 'WAVs/' + filename + '.wav'

    # Load the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # If stereo, convert to mono by averaging the channels
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    # Calculate the spectrogram with a larger FFT window size
    nperseg = 4094  # Larger window size for better frequency resolution
    noverlap = nperseg // 1.5 #Strange grey bars appear for values greater than 1.5

    frequencies, times, Sxx = spectrogram(data, fs=sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap)

    # Convert the spectrogram (power spectral density) to decibels
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Adding a small number to avoid log(0)

    # Calculate the duration of a 32nd note in seconds
    beats_per_second = bpm / 60
    seconds_per_beat = 1 / beats_per_second
    seconds_per_32nd_note = seconds_per_beat / 8  # 32nd note duration

    # Determine the number of time slices for each 32nd note duration
    num_slices = int(np.ceil(times[-1] / seconds_per_32nd_note))

    # List to store the average values of each vertical slice
    avg_slices = []

    # Iterate over each 32nd note slice
    for I in range(num_slices):
        # Determine the start and end time for this slice
        start_time = I * seconds_per_32nd_note
        end_time = (I + 1) * seconds_per_32nd_note

        # Find the indices in the time array that correspond to this slice
        start_idx = np.searchsorted(times, start_time)
        end_idx = np.searchsorted(times, end_time)

        # Get the slice of the spectrogram for this time period
        slice_Sxx_dB = Sxx_dB[:, start_idx:end_idx]

        # Calculate the average value of each vertical pixel in this slice
        avg_values = np.mean(slice_Sxx_dB, axis=1)
        avg_slices.append(avg_values)

    # Convert the list of average slices to a numpy array for further processing
    avg_slices_array = np.array(avg_slices)
    
    return avg_slices_array

def process_wav_file_for_prediction(model, filename, bpm, trim_length):
    # Generate spectrogram slices
    spectrogram_slices = create_amplitude_tensors(filename, bpm)

    # Reshape the slices for the model (add channel dimension)
    spectrogram_slices = spectrogram_slices.reshape((spectrogram_slices.shape[0], spectrogram_slices.shape[1], 1))

    # Predict using the model
    predictions = model.predict(spectrogram_slices)

    # Normalize predictions to the range 0-255 for image representation
    predictions_normalized = (predictions - np.min(predictions)) / np.ptp(predictions) * 255
    predictions_normalized = predictions_normalized.astype(np.uint8)

    predictions_trimmed = predictions_normalized[:trim_length]

    threshold = 220

    for i in range(len(predictions)):
        for j in range(len(predictions[0])):
            if predictions[i][j] <= threshold: predictions[i][j] = 0

    # Create and save the image
    image = Image.fromarray(predictions_trimmed.T)  # Transpose to match desired format
    image.save('OutputImages/' + filename + '_output.png')

    return predictions




# ----------------------------------------------------------------------------------------------------------

# ---MIDI PROCESSING---

def load_midi(file_path):
    """Load the MIDI file and return the messages with their cumulative times."""
    midi_file = mido.MidiFile(file_path)
    messages_with_time = []

    # Initialize the current time
    current_time = 0

    for message in midi_file:
        # Increment the current time by the time of the current message
        current_time += message.time
        # Append the message with the cumulative time to the list
        messages_with_time.append((current_time, message))

    return messages_with_time

def get_note_periods(messages_with_time):
    """Get the time periods for each note."""
    note_periods = []
    notes_on = {}

    for time, message in messages_with_time:
        if message.type == 'note_on' and message.velocity > 0:
            if message.note not in notes_on:
                notes_on[message.note] = []
            notes_on[message.note].append(time)
        elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
            if message.note in notes_on and notes_on[message.note]:
                start_time = notes_on[message.note].pop()
                note_periods.append((message.note, start_time, time))

    # If there are notes that were not turned off, handle them appropriately
    for note, times in notes_on.items():
        for start_time in times:
            note_periods.append((note, start_time, messages_with_time[-1][0]))

    return note_periods

def create_note_dict(note_periods):
    """Create a dictionary of note periods."""
    note_dict = {}
    note_id = 0

    for note, start_time, end_time in note_periods:
        note_dict[note_id] = [note, (start_time, end_time)]
        note_id += 1

    return note_dict

def get_notes_in_32nd_period(note_dict, start_time, end_time):
    """Get one-hot encoded notes for a specific 32nd-note period."""
    notes_playing = set()
    period_duration = end_time - start_time
    threshold = period_duration / 2

    for note_info in note_dict.values():
        note, (note_start, note_end) = note_info
        overlap_start = max(note_start, start_time)
        overlap_end = min(note_end, end_time)
        overlap_duration = overlap_end - overlap_start

        if overlap_duration > threshold:
            notes_playing.add(note)

    # Create a one-hot encoded array for notes 40 to 88
    one_hot_array = [0] * (88 - 40 + 1)
    for note in notes_playing:
        if 40 <= note <= 88:
            one_hot_array[note - 40] = 1

    return one_hot_array

def get_all_32nd_note_periods(note_dict, start_time, end_time, period_duration):
    """Generate one-hot encoded arrays for all 32nd-note periods."""
    current_time = start_time
    periods = []

    while current_time < end_time:
        next_time = current_time + period_duration
        one_hot_array = get_notes_in_32nd_period(note_dict, current_time, next_time)
        periods.append(one_hot_array)
        current_time = next_time

    return periods

def create_midi_tensors(filename):
    """Main function to load the MIDI file and get one-hot encoded note periods."""

    file_path="MIDIs/"+filename+".mid"
    messages_with_time = load_midi(file_path)
    note_periods = get_note_periods(messages_with_time)
    note_dict = create_note_dict(note_periods)
        
    one_hot_encoded_periods = get_all_32nd_note_periods(note_dict, 0, 8, 0.0625)

    one_hot_encoded_periods = np.array(one_hot_encoded_periods).astype(np.uint8)
    # print(one_hot_encoded_periods[0])
    
    image = Image.fromarray(one_hot_encoded_periods.T[::-1] * 255)  # Transpose to match desired format
    image.save('OutputImages/' + filename + '_midi.png')

    return one_hot_encoded_periods

# Load the trained model
model = tf.keras.models.load_model('*saved_tf_models/BasicConvGuitarNotePredictor2.keras')

# Process the WAV file and generate the output image
filename = 'aavBuDcvHLUfUHTanMUQ'
bpm = 120
midi = create_midi_tensors(filename)
# print(midi[0])
len_midi_array = len(midi[0])
predictions = process_wav_file_for_prediction(model, filename, bpm, len_midi_array)