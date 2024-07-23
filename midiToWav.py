import subprocess

def convert(filename, soundfont_path):
    # Paths
    midi_file_path = 'MIDIs/'+filename+'.mid'
    # soundfont_path = 'FluidR3_GM.sf2'  
    output_wav_path = 'WAVs/'+filename+'.wav'

    # FluidSynth command
    command = [
        'fluidsynth',
        '-ni',
        soundfont_path,
        midi_file_path,
        '-F', output_wav_path,
        '-r', '44100'
    ]

    # Run the FluidSynth command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check result
    if result.returncode == 0:
        print('MIDI has been successfully converted to WAV.')
    else:
        print('Error converting MIDI to WAV:')
        print(result.stderr)
