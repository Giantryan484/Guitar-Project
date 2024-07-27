# Pre-Processing
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, filtfilt

# Resampling WAVs and removing
from pydub import AudioSegment
import os

# Post-Processing
from itertools import combinations, product

# <---Pre-Processing and Prediction--->

# ---WAV PROCESSING---

def bandpass_filter(data, lowcut, highcut, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Wave file path -> slices of amplitudes
def create_amplitude_tensors(filename, bpm=120):
    wav_file = filename

    # Load the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # If stereo, convert to mono by averaging the channels
    if len(data.shape) == 2:
        data = data.mean(axis=1)

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

def change_sample_rate(file_path):
    sound = AudioSegment.from_file(file_path)
    sound = sound.set_frame_rate(44100)
    new_filepath = "tmp/"+file_path
    sound.export(new_filepath, format="wav")

    return new_filepath

# Filename -> predictions
def process_wav_file_and_predict(model, filename, bpm):
    # Generate spectrogram slices
    
    resampled_wav_filename = change_sample_rate(filename)
    spectrogram_slices = create_amplitude_tensors(resampled_wav_filename, bpm)
    os.remove(resampled_wav_filename)

    # Reshape the slices for the model (add channel dimension)
    spectrogram_slices = spectrogram_slices.reshape(
        (spectrogram_slices.shape[0], spectrogram_slices.shape[1], 1)
    )

    # Predict using the model
    predictions = model.predict(spectrogram_slices, verbose=0)

    return predictions

def cleanup_and_aggregate(model, data, window_size=10, threshold=0.5):    
    num_samples, width = data.shape
    predictions = np.zeros((num_samples, width))
    counts = np.zeros((num_samples, width))

    for i in range(num_samples - window_size + 1):
        window = data[i:i+window_size].flatten().reshape(1, -1)
        pred = model.predict(window, verbose=0)
        pred = pred.reshape(window_size, width)
        predictions[i:i+window_size] += pred
        counts[i:i+window_size] += 1

    averaged_predictions = predictions / counts
    final_predictions = (averaged_predictions >= threshold).astype(int)
    return final_predictions

# Generate final output array
# final_output = predict_and_aggregate(model, input_data, window_size)

# <---END Pre-Processing and Prediction--->



# <---Post-Processing (Transcription to TABs)--->
midi_to_note = ['E|2', 'F|2', 'F#|2', 'G|2', 'G#|2', 'A|2', 'A#|2', 'B|2', 'C|3', 'C#|3', 'D|3', 'D#|3', 'E|3', 'F|3', 'F#|3', 'G|3', 'G#|3', 'A|3', 'A#|3', 'B|3', 'C|4', 'C#|4', 'D|4', 'D#|4', 'E|4', 'F|4', 'F#|4', 'G|4', 'G#|4', 'A|4', 'A#|4', 'B|4', 'C|5', 'C#|5', 'D|5', 'D#|5', 'E|5', 'F|5', 'F#|5', 'G|5', 'G#|5', 'A|5', 'A#|5', 'B|5', 'C|6', 'C#|6', 'D|6', 'D#|6', 'E|6']
midi_to_sfret = {'E|2': ['E|0'], 'F|2': ['E|1'], 'F#|2': ['E|2'], 'G|2': ['E|3'], 'G#|2': ['E|4'], 'A|2': ['E|5', 'A|0'], 'A#|2': ['E|6', 'A|1'], 'B|2': ['E|7', 'A|2'], 'C|3': ['E|8', 'A|3'], 'C#|3': ['E|9', 'A|4'], 'D|3': ['E|10', 'A|5', 'D|0'], 'D#|3': ['E|11', 'A|6', 'D|1'], 'E|3': ['E|12', 'A|7', 'D|2'], 'F|3': ['E|13', 'A|8', 'D|3'], 'F#|3': ['E|14', 'A|9', 'D|4'], 'G|3': ['E|15', 'A|10', 'D|5', 'G|0'], 'G#|3': ['E|16', 'A|11', 'D|6', 'G|1'], 'A|3': ['E|17', 'A|12', 'D|7', 'G|2'], 'A#|3': ['E|18', 'A|13', 'D|8', 'G|3'], 'B|3': ['E|19', 'A|14', 'D|9', 'G|4', 'b|0'], 'C|4': ['E|20', 'A|15', 'D|10', 'G|5', 'b|1'], 'C#|4': ['E|21', 'A|16', 'D|11', 'G|6', 'b|2'], 'D|4': ['E|22', 'A|17', 'D|12', 'G|7', 'b|3'], 'D#|4': ['E|23', 'A|18', 'D|13', 'G|8', 'b|4'], 'E|4': ['A|19', 'D|14', 'G|9', 'b|5', 'e|0'], 'F|4': ['A|20', 'D|15', 'G|10', 'b|6', 'e|1'], 'F#|4': ['A|21', 'D|16', 'G|11', 'b|7', 'e|2'], 'G|4': ['A|22', 'D|17', 'G|12', 'b|8', 'e|3'], 'G#|4': ['A|23', 'D|18', 'G|13', 'b|9', 'e|4'], 'A|4': ['D|19', 'G|14', 'b|10', 'e|5'], 'A#|4': ['D|20', 'G|15', 'b|11', 'e|6'], 'B|4': ['D|21', 'G|16', 'b|12', 'e|7'], 'C|5': ['D|22', 'G|17', 'b|13', 'e|8'], 'C#|5': ['D|23', 'G|18', 'b|14', 'e|9'], 'D|5': ['G|19', 'b|15', 'e|10'], 'D#|5': ['G|20', 'b|16', 'e|11'], 'E|5': ['G|21', 'b|17', 'e|12'], 'F|5': ['G|22', 'b|18', 'e|13'], 'F#|5': ['G|23', 'b|19', 'e|14'], 'G|5': ['b|20', 'e|15'], 'G#|5': ['b|21', 'e|16'], 'A|5': ['b|22', 'e|17'], 'A#|5': ['b|23', 'e|18'], 'B|5': ['e|19'], 'C|6': ['e|20'], 'C#|6': ['e|21'], 'D|6': ['e|22'], 'D#|6': ['e|23'], 'E|6': ['e|24']}
sfret_to_note = {'E|0': 0, 'E|1': 1, 'E|2': 2, 'E|3': 3, 'E|4': 4, 'E|5': 5, 'A|0': 5, 'E|6': 6, 'A|1': 6, 'E|7': 7, 'A|2': 7, 'E|8': 8, 'A|3': 8, 'E|9': 9, 'A|4': 9, 'E|10': 10, 'A|5': 10, 'D|0': 10, 'E|11': 11, 'A|6': 11, 'D|1': 11, 'E|12': 12, 'A|7': 12, 'D|2': 12, 'E|13': 13, 'A|8': 13, 'D|3': 13, 'E|14': 14, 'A|9': 14, 'D|4': 14, 'E|15': 15, 'A|10': 15, 'D|5': 15, 'G|0': 15, 'E|16': 16, 'A|11': 16, 'D|6': 16, 'G|1': 16, 'E|17': 17, 'A|12': 17, 'D|7': 17, 'G|2': 17, 'E|18': 18, 'A|13': 18, 'D|8': 18, 'G|3': 18, 'E|19': 19, 'A|14': 19, 'D|9': 19, 'G|4': 19, 'b|0': 19, 'E|20': 20, 'A|15': 20, 'D|10': 20, 'G|5': 20, 'b|1': 20, 'E|21': 21, 'A|16': 21, 'D|11': 21, 'G|6': 21, 'b|2': 21, 'E|22': 22, 'A|17': 22, 'D|12': 22, 'G|7': 22, 'b|3': 22, 'E|23': 23, 'A|18': 23, 'D|13': 23, 'G|8': 23, 'b|4': 23, 'A|19': 24, 'D|14': 24, 'G|9': 24, 'b|5': 24, 'e|0': 24, 'A|20': 25, 'D|15': 25, 'G|10': 25, 'b|6': 25, 'e|1': 25, 'A|21': 26, 'D|16': 26, 'G|11': 26, 'b|7': 26, 'e|2': 26, 'A|22': 27, 'D|17': 27, 'G|12': 27, 'b|8': 27, 'e|3': 27, 'A|23': 28, 'D|18': 28, 'G|13': 28, 'b|9': 28, 'e|4': 28, 'D|19': 29, 'G|14': 29, 'b|10': 29, 'e|5': 29, 'D|20': 30, 'G|15': 30, 'b|11': 30, 'e|6': 30, 'D|21': 31, 'G|16': 31, 'b|12': 31, 'e|7': 31, 'D|22': 32, 'G|17': 32, 'b|13': 32, 'e|8': 32, 'D|23': 33, 'G|18': 33, 'b|14': 33, 'e|9': 33, 'G|19': 34, 'b|15': 34, 'e|10': 34, 'G|20': 35, 'b|16': 35, 'e|11': 35, 'G|21': 36, 'b|17': 36, 'e|12': 36, 'G|22': 37, 'b|18': 37, 'e|13': 37, 'G|23': 38, 'b|19': 38, 'e|14': 38, 'b|20': 39, 'e|15': 39, 'b|21': 40, 'e|16': 40, 'b|22': 41, 'e|17': 41, 'b|23': 42, 'e|18': 42, 'e|19': 43, 'e|20': 44, 'e|21': 45, 'e|22': 46, 'e|23': 47, 'e|24': 48}

def get_distance(sfret1, sfret2): # string/fret pair
    fret1 = int(sfret1.split("|")[1])
    fret2 = int(sfret2.split("|")[1])
    
    if fret1 == 0 or fret2 == 0: # Open string, super simple
        return 0

    distance_weights = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136, 153, 171, 190, 210, 231, 253, 276]
    
    return distance_weights[abs(fret1-fret2)]

def check_for_duplicate_strings(sfret_list):
    string_list = [sfret.split("|")[0] for sfret in sfret_list]
    return (len(string_list) != len(set(string_list)))

def generate_2_length_combinations(values_list):
    all_combinations = list(combinations(values_list, 2))
    return all_combinations

def get_score(sfret_list, prev_center=0):
    # if check_for_duplicate_strings(sfret_list): # somehow only check notes that are actually on top of each other to avoid close-together notes being forced apart
    #     return 100000
    sfret_combinations = generate_2_length_combinations(sfret_list)
    score = sum([get_distance(sfret_pair[0], sfret_pair[1]) for sfret_pair in sfret_combinations])
    if not sfret_combinations and int(sfret_list[0].split("|")[1]) == 0:
        score = -1
    # print("current combo: ", sfret_list)
    # print("before weighting:", score)
    
    center = get_center(sfret_list)
    # print("center:", center)
    # if prev_center: # Prefer TABs that are closer together in general
    center_diff = abs(prev_center - center)
    score += (2*center_diff)
    
    # print("after weighting:", score)
    
    return score

def generate_combinations(dictionary, keys):
    # Extract the list of values for each key
    values_lists = [dictionary[key] for key in keys]
    
    # Generate all combinations of the values
    all_combinations = list(product(*values_lists))
    
    return all_combinations

def get_center(sfret_list):
    sfret_list_no_zeros = [int(sfret.split("|")[1]) for sfret in sfret_list if int(sfret.split("|")[1]) != 0]
    if len(sfret_list_no_zeros) == 0: # avoid divide by 0
        return 0
    return round(sum(sfret_list_no_zeros) / len(sfret_list_no_zeros))

# sfrets = ["E|0", "A|0", "D|22", "D|22", "b|22", "e|22"]

def process_slices(midi_slices):
    prev_center = 0
    output_tabs = []
    old_slices = midi_slices

    for slice_idx in range(0, len(midi_slices), 2):  # Process 16th note slices from 32nd-note slices
        current_slice = midi_slices[slice_idx]
        notes_being_pressed = [midi_to_note[i] for i, val in enumerate(current_slice) if type(val) != str and val > 0.5]
        existing_fingerings = [val for val in current_slice if type(val) == str]
        
        
        if not notes_being_pressed:
            output_tabs.append([])
            continue
        
        possible_sfrets = generate_combinations(midi_to_sfret, notes_being_pressed)
        best_score = float('inf')
        best_combination = []
        
        # print("goal:", notes_being_pressed)

        for combination in possible_sfrets:
            combination = list(combination)
            # if not check_for_duplicate_strings(combination + existing_fingerings):
            if not check_for_duplicate_strings(combination):
                score = get_score(combination + existing_fingerings, prev_center) # account for current notes, center
                # print(combination, score)
                if score < best_score:
                    # print("new best found:", combination, score)
                    best_score = score
                    best_combination = combination

        prev_center = get_center(best_combination + existing_fingerings) - 1
        output_tabs.append(best_combination)
        
        for sfret in best_combination:
            index = sfret_to_note[sfret]
            current_idx = slice_idx
            while current_idx < len(midi_slices) and midi_slices[current_idx][index] > 0.5:
                midi_slices[current_idx][index] = sfret
                current_idx += 1
    # print(midi_slices)
    return output_tabs

def TABs_from_output(output):
    # Initialize TABs
    TABs = {
        "E": [],
        "A": [],
        "D": [],
        "G": [],
        "b": [],
        "e": []
    }
    # Plot notes on the TABs
    for i in range(len(output)): # change to index
        # for string in sfrets, add (i, fret) to TABs
        notes_slice = output[i]
        for sfret in notes_slice:
            string = sfret.split("|")[0]
            fret = sfret.split("|")[1]
            TABs[string].append((i, fret))

    # Format TABs with measures
    measures = len(output) // 16 + 1
    formatted_tabs = {
        "E": [],
        "A": [],
        "D": [],
        "G": [],
        "b": [],
        "e": []
    }
    for string in TABs: # for each string
        current_measure = []
        for i in range(measures * 16): #for every index (0-15) for 1 measure
            if (i) % 16 == 0: # if divisible by 16 (start of measure)
                if current_measure:
                    formatted_tabs[string].append("".join(current_measure) + "-|-") # adds divider between rows
                    current_measure = []
            note_at_position = "--" # default value to add is empty
            for pos, fret in TABs[string]:
                if pos == i:
                    if len(fret) == 2:
                        note_at_position = fret
                    else:
                        note_at_position = fret+"-" # if present change value to add
            current_measure.append(note_at_position)
        if current_measure:
            formatted_tabs[string].append("".join(current_measure) + "-|")

    # Print the TABs
    print("e|-" + "".join(formatted_tabs["e"]))
    print("B|-" + "".join(formatted_tabs["b"]))
    print("G|-" + "".join(formatted_tabs["G"]))
    print("D|-" + "".join(formatted_tabs["D"]))
    print("A|-" + "".join(formatted_tabs["A"]))
    print("E|-" + "".join(formatted_tabs["E"]))

# <---END Post-Processing--->

def main(filename, bpm):
    model = tf.keras.models.load_model("saved_tf_models\BasicConvGuitarNotePredictor(512_input).keras") # model path
    cleanup_model = tf.keras.models.load_model("saved_tf_models\DataCleaner(WindowSize10).h5") # cleanup model path
    
    predictions = process_wav_file_and_predict(model, filename, bpm)
    cleaned_predictions = cleanup_and_aggregate(cleanup_model, predictions)
    TABs_from_output(process_slices(cleaned_predictions.tolist()))


main("test.wav", 80)
    
    
    