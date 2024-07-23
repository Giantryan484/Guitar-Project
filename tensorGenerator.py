from scipy.io import wavfile
from scipy.signal import spectrogram
from PIL import Image
import numpy as np
import mido
from scipy.signal import spectrogram, butter, filtfilt



# ---WAV PROCESSING---

def generate_image_from_array(array, output_file):
    # Convert the array to a numpy array if it isn't already
    array = np.array(array)
    
    # Normalize the array values to be between 0 and 255
    normalized_array = normalize_array(array)
    
    # Create an image from the normalized array
    # Transpose the array to have inner arrays as vertical columns
    image = Image.fromarray(normalized_array.T)
    
    # Save the image
    image.save(output_file)

def normalize_array(array):
    if np.ptp(array) == 0:  # Check if the array has zero range
        return np.zeros(array.shape, dtype=np.uint8)
    normalized_array = 255 * (array - np.min(array)) / np.ptp(array)
    return normalized_array.astype(np.uint8)

def bandpass_filter(data, lowcut, highcut, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def create_amplitude_tensors(filename, bpm):
    wav_file = 'WAVs/' + filename + '.wav'
    output_file = 'Spectrograms/' + filename + '.png'

    # Load the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # If stereo, convert to mono by averaging the channels
    if len(data.shape) == 2:
        data = data.mean(axis=1)

        # Define the band-pass filter

    # Apply the band-pass filter
    lowcut = 70  # E2 frequency in Hz
    highcut = 1700  # E6 frequency in Hz
    data = bandpass_filter(data, lowcut, highcut, sample_rate)

    # Calculate the spectrogram with a larger FFT window size
    nperseg = 4094  # Larger window size for better frequency resolution
    noverlap = nperseg // 1.5 #Strange grey bars appear for values greater than 1.5

    frequencies, times, Sxx = spectrogram(data, fs=sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap)

    # Convert the spectrogram (power spectral density) to decibels
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Adding a small number to avoid log(0)

    Sxx_dB = Sxx_dB[:][:512]

    # Normalize the values between 0 and 255
    img_array = np.uint8(255 * (Sxx_dB - np.min(Sxx_dB)) / np.ptp(Sxx_dB))
    
    # Convert to Image and save as PNG
    image = Image.fromarray(img_array)
    image.save(output_file)

    # Define the frequencies of guitar notes from E2 to E6
    # guitar_note_frequencies = [
    #     ('E2', 87.31),
    #     ('F2', 92.50),
    #     ('F#2', 98.00),
    #     ('G2', 103.83),
    #     ('G#2', 110.00),
    #     ('A2', 116.54),
    #     ('A#2', 123.47),
    #     ('B2', 130.81),
    #     ('C3', 138.59),
    #     ('C#3', 146.83),
    #     ('D3', 155.56),
    #     ('D#3', 164.81),
    #     ('E3', 174.61),
    #     ('F3', 185.00),
    #     ('F#3', 196.00),
    #     ('G3', 207.65),
    #     ('G#3', 220.00),
    #     ('A3', 233.08),
    #     ('A#3', 246.94),
    #     ('B3', 261.63),
    #     ('C4', 277.18),
    #     ('C#4', 293.66),
    #     ('D4', 311.13),
    #     ('D#4', 329.63),
    #     ('E4', 349.23),
    #     ('F4', 369.99),
    #     ('F#4', 391.99),
    #     ('G4', 415.30),
    #     ('G#4', 440.00),
    #     ('A4', 466.16),
    #     ('A#4', 493.88),
    #     ('B4', 523.25),
    #     ('C5', 554.37),
    #     ('C#5', 587.33),
    #     ('D5', 622.25),
    #     ('D#5', 659.26),
    #     ('E5', 698.46),
    #     ('F5', 739.99),
    #     ('F#5', 783.99),
    #     ('G5', 830.61),
    #     ('G#5', 880.00),
    #     ('A5', 932.33),
    #     ('A#5', 987.77),
    #     ('B5', 1046.50),
    #     ('C6', 1108.73),
    #     ('C#6', 1174.66),
    #     ('D6', 1244.51),
    #     ('D#6', 1318.51),
    #     ('E6', 1396.91)
    # ]

    # Calculate the duration of a 32nd note in seconds
    beats_per_second = bpm / 60
    seconds_per_beat = 1 / beats_per_second
    seconds_per_32nd_note = seconds_per_beat / 8  # 32nd note duration

    # Determine the number of time slices for each 32nd note duration
    num_slices = int(np.ceil(times[-1] / seconds_per_32nd_note))

    # # Initialize 2D array to store average amplitudes for each note and time slice
    # note_amplitudes_array = np.zeros((num_slices, len(guitar_note_frequencies)))

    # # Calculate average amplitude for each note across each time slice
    # for i, (note, target_freq) in enumerate(guitar_note_frequencies):
    #     freq_bin_width = frequencies[1] - frequencies[0]
    #     low = target_freq - freq_bin_width / 2
    #     high = target_freq + freq_bin_width / 2
    #     indices = np.where((frequencies >= low) & (frequencies <= high))[0]
    #     if indices.size > 0:
    #         for j in range(num_slices):
    #             start_time = j * seconds_per_32nd_note
    #             end_time = start_time + seconds_per_32nd_note
    #             time_indices = np.where((times >= start_time) & (times < end_time))[0]
    #             if time_indices.size > 0:
    #                 avg_amplitude = np.mean(Sxx_dB[indices[:, np.newaxis], time_indices], axis=1).mean()
    #                 note_amplitudes_array[j, i] = avg_amplitude
    
    # # generate_image_from_array(note_amplitudes_array)
    # return note_amplitudes_array


    # List to store the average values of each vertical slice
    avg_slices = []

    # Iterate over each 32nd note slice
    for i in range(num_slices):
        # Determine the start and end time for this slice
        start_time = i * seconds_per_32nd_note
        end_time = (i + 1) * seconds_per_32nd_note

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

def create_midi_tensors(file_path):
    """Main function to load the MIDI file and get one-hot encoded note periods."""

    file_path="MIDIs/"+file_path+".mid"
    messages_with_time = load_midi(file_path)
    note_periods = get_note_periods(messages_with_time)
    note_dict = create_note_dict(note_periods)
        
    one_hot_encoded_periods = get_all_32nd_note_periods(note_dict, 0, 8, 0.0625)
    # print(one_hot_encoded_periods[2])
    
    return np.array(one_hot_encoded_periods)




# ----------------------------------------------------------------------------------------------------------

# Main function:

def build_tensors(filename, bpm):
    midi_list = create_midi_tensors(filename)
    amplitudes_list = create_amplitude_tensors(filename, bpm)
    if len(midi_list) > len(amplitudes_list):
        print("Error: amplitude list smaller than midi list for unknown reason, aborting...")
        return []
    master_list = [[amplitudes_list[i], midi_list[i]] for i in range(len(midi_list))]
    return master_list