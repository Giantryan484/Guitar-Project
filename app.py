import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.signal import butter, filtfilt, spectrogram
from scipy.io import wavfile
import pygame
import time

# Bandpass filter function
def bandpass_filter(data, lowcut, highcut, sample_rate, order=5):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Function to create amplitude tensors
def create_amplitude_tensors(filename, tap_times):
    sample_rate, data = wavfile.read(filename)

    # If stereo, convert to mono by averaging the channels
    if len(data.shape) == 2:
        data = data.mean(axis=1)

    # Apply the band-pass filter
    lowcut = 70  # E2 frequency in Hz
    highcut = 1700  # E6 frequency in Hz
    data = bandpass_filter(data, lowcut, highcut, sample_rate)

    # Calculate the spectrogram with a larger FFT window size
    nperseg = 4094  # Larger window size for better frequency resolution
    noverlap = int(nperseg / 1.5)

    frequencies, times, Sxx = spectrogram(data, fs=sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap)

    # Convert the spectrogram (power spectral density) to decibels
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Adding a small number to avoid log(0)
    Sxx_dB = Sxx_dB[:][:512]

    avg_slices = []

    # Process the spectrogram based on tap times
    for i in range(len(tap_times) - 1):
        start_time = tap_times[i]
        end_time = tap_times[i + 1]

        # Determine the duration of a quarter note
        quarter_note_duration = end_time - start_time
        seconds_per_32nd_note = quarter_note_duration / 8  # 32nd note duration

        start_idx = np.searchsorted(times, start_time)
        end_idx = np.searchsorted(times, end_time)

        slice_Sxx_dB = Sxx_dB[:, start_idx:end_idx]

        # Divide the slice into equal 32nd note durations
        num_slices = int((end_time - start_time) / seconds_per_32nd_note)
        for j in range(num_slices):
            slice_start_idx = int(j * seconds_per_32nd_note * sample_rate / nperseg)
            slice_end_idx = int((j + 1) * seconds_per_32nd_note * sample_rate / nperseg)

            avg_values = np.mean(slice_Sxx_dB[:, slice_start_idx:slice_end_idx], axis=1)
            avg_slices.append(avg_values)

    avg_slices_array = np.array(avg_slices)
    return avg_slices_array

# GUI class for the application
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Tempo Tapper")

        self.filename = None
        self.tap_times = []

        self.load_button = tk.Button(root, text="Load WAV File", command=self.load_file)
        self.load_button.pack()

        self.record_button = tk.Button(root, text="Record Taps", command=self.record_taps, state=tk.DISABLED)
        self.record_button.pack()

        self.process_button = tk.Button(root, text="Process Spectrogram", command=self.process_spectrogram, state=tk.DISABLED)
        self.process_button.pack()

        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.pack()

    def load_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if self.filename:
            self.record_button.config(state=tk.NORMAL)

    def record_taps(self):
        self.tap_times = []
        pygame.mixer.init()
        pygame.mixer.music.load(self.filename)
        pygame.mixer.music.play()

        self.start_time = time.time()
        self.root.bind('<space>', self.record_tap)
        self.root.after(1000, self.check_music)

    def record_tap(self, event):
        tap_time = time.time() - self.start_time
        self.tap_times.append(tap_time)

    def check_music(self):
        if not pygame.mixer.music.get_busy():
            self.root.unbind('<space>')
            self.process_button.config(state=tk.NORMAL)
        else:
            self.root.after(1000, self.check_music)

    def process_spectrogram(self):
        avg_slices = create_amplitude_tensors(self.filename, self.tap_times)
        print(avg_slices)  # You can change this to save or visualize the output

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
