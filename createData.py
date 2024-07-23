import random
import string
import os
import midiGenterator
import midiToWav
import tensorGenerator
from tensorflow import train, io

# Define the paths
soundfont_dir = 'SoundFonts'
output_dir = 'GeneratedData'
os.makedirs(output_dir, exist_ok=True)

# Helper functions for TFRecord
def _float_feature(value):
    return train.Feature(float_list=train.FloatList(value=value))

def _int64_feature(value):
    return train.Feature(int64_list=train.Int64List(value=value))

def serialize_example(spectrogram, notes):
    feature = {
        'spectrogram': _float_feature(spectrogram),
        'notes': _int64_feature(notes)
    }
    example_proto = train.Example(features=train.Features(feature=feature))
    return example_proto.SerializeToString()

# Iterate over all .sf2 files in the soundFont directory
for soundfont in os.listdir(soundfont_dir):
    if soundfont.endswith('.sf2'):
        soundfont_fp = os.path.join(soundfont_dir, soundfont)
        soundfont_name = os.path.splitext(os.path.basename(soundfont_fp))[0]

        
        for _ in range(20):  # Adjust the range to control the number of samples per soundfont
            filename = ''.join(random.choices(string.ascii_letters, k=20))
            midiGenterator.generate_random_basic(filename)
            midiToWav.convert(filename, soundfont_fp)
            
            tensors = tensorGenerator.build_tensors(filename, 120)

            for tensor in tensors:
                if len(tensor[0]) != 2048 or len(tensor[1]) != 49:
                    print("Found incorrect shape in file: "+filename)
            
            # Save tensors to TFRecord
            tfrecord_filename = os.path.join(output_dir, f'{soundfont_name}_{filename}.tfrecord')
            with io.TFRecordWriter(tfrecord_filename) as writer:
                for tensor in tensors:
                    spectrogram, notes = tensor  # assuming tensor is a tuple (spectrogram, notes)
                    tf_example = serialize_example(spectrogram, notes)
                    writer.write(tf_example)

            print(f'Saved {tfrecord_filename}')