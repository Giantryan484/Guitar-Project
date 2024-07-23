import random
import string
import os
import midiGenterator
import midiToWav
import tensorGenerator
from tensorflow import train, io

import os
import random
import numpy as np

def get_random_filename(folder_path):
    """Return a random filename from the given folder."""
    filenames = os.listdir(folder_path)
    return os.path.join(folder_path, random.choice(filenames))

# Define the paths
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

for filename in os.listdir("MIDIs"):
    filename = filename[:-4]

    tensors = tensorGenerator.build_tensors(filename, 120)

    for tensor in tensors:
        if len(tensor[0]) != 512 or len(tensor[1]) != 49:
            print("Found incorrect shape in file: "+filename)

    # Save tensors to TFRecord
    tfrecord_filename = os.path.join(output_dir, f'{filename}.tfrecord')
    with io.TFRecordWriter(tfrecord_filename) as writer:
        for tensor in tensors:
            spectrogram, notes = tensor  # assuming tensor is a tuple (spectrogram, notes)
            tf_example = serialize_example(spectrogram, notes)
            writer.write(tf_example)

    print(f'Saved {tfrecord_filename}')