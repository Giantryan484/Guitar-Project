import pyaudio
import numpy as np
import sounddevice as sd
from scipy.signal import spectrogram
from PIL import Image, ImageTk
import tkinter as tk
import threading
import time

# Parameters
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate
NPERSEG = 1024  # Segment length for spectrogram
NOVERLAP = 512  # Overlap between segments
IMG_WIDTH = 600  # Image width
IMG_HEIGHT = 400  # Image height

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Initialize tkinter window
root = tk.Tk()
root.title("Live Spectrogram")
canvas = tk.Canvas(root, width=IMG_WIDTH, height=IMG_HEIGHT)
canvas.pack()

# Global variables
spectrogram_image = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 'black')
spectrogram_photo = ImageTk.PhotoImage(spectrogram_image)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=spectrogram_photo)
recording = True
input_mode = tk.StringVar(value="Microphone")  # Toggle between 'Microphone' and 'Output'

def capture_audio_from_microphone():
    global spectrogram_image, spectrogram_photo
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    while recording and input_mode.get() == "Microphone":
        try:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            update_spectrogram(data)
        except IOError as e:
            print(f"Error reading audio stream: {e}")
    stream.stop_stream()
    stream.close()

def capture_audio_from_output():
    global spectrogram_image, spectrogram_photo
    output_device_index = 2  # Change this to the index of your system output device

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        # Play silence to keep the stream open
        outdata.fill(0)
        update_spectrogram(indata[:, 0])
    
    with sd.Stream(callback=callback, channels=1, samplerate=RATE, blocksize=CHUNK, device=(None, output_device_index)):
        while recording and input_mode.get() == "Output":
            sd.sleep(100)

def update_spectrogram(data):
    global spectrogram_image, spectrogram_photo
    if np.all(data == 0):
        return
    frequencies, times, Sxx = spectrogram(data, fs=RATE, window='hann', nperseg=NPERSEG, noverlap=NOVERLAP)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Convert to decibels
    img_array = np.uint8(255 * (Sxx_dB - np.nanmin(Sxx_dB)) / np.ptp(Sxx_dB))  # Normalize

    # Resize the slice to match the image height
    slice_image = Image.fromarray(img_array).resize((1, IMG_HEIGHT))

    # Scroll the spectrogram image to the left
    spectrogram_image.paste(spectrogram_image.crop((1, 0, IMG_WIDTH, IMG_HEIGHT)), (-1, 0))
    spectrogram_image.paste(slice_image, (IMG_WIDTH - 1, 0))

    # Update the display
    spectrogram_photo = ImageTk.PhotoImage(spectrogram_image)
    canvas.itemconfig(image_on_canvas, image=spectrogram_photo)
    root.update_idletasks()

def start_recording():
    global recording
    recording = True
    if input_mode.get() == "Microphone":
        audio_thread = threading.Thread(target=capture_audio_from_microphone)
    else:
        audio_thread = threading.Thread(target=capture_audio_from_output)
    audio_thread.start()

def stop_recording():
    global recording
    recording = False

def toggle_input_mode():
    if input_mode.get() == "Microphone":
        input_mode.set("Output")
    else:
        input_mode.set("Microphone")

def on_closing():
    stop_recording()
    root.destroy()

# Add buttons and toggle for input mode
start_button = tk.Button(root, text="Start", command=start_recording)
start_button.pack(side=tk.LEFT)
stop_button = tk.Button(root, text="Stop", command=stop_recording)
stop_button.pack(side=tk.LEFT)
toggle_button = tk.Button(root, text="Toggle Input", command=toggle_input_mode)
toggle_button.pack(side=tk.LEFT)

# Display input mode
input_mode_label = tk.Label(root, textvariable=input_mode)
input_mode_label.pack(side=tk.LEFT)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()