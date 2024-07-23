import tensorflow as tf
# import os
# import numpy as np

# # Parsing function
# def _parse_function(proto):
#     keys_to_features = {
#         'spectrogram': tf.io.FixedLenFeature([2048], tf.float32),
#         'notes': tf.io.FixedLenFeature([49], tf.int64)
#     }
#     parsed_features = tf.io.parse_single_example(proto, keys_to_features)
#     return parsed_features['spectrogram'], parsed_features['notes']

# # Augmentation functions
# def add_noise(spectrogram, noise_factor=0.005):
#     noise = np.random.randn(*spectrogram.shape) * noise_factor
#     return spectrogram + noise

# def amplitude_scaling(spectrogram, scale_range=(0.9, 1.1)):
#     scale = np.random.uniform(scale_range[0], scale_range[1])
#     return spectrogram * scale

# def augment_data(spectrogram, notes):
#     spectrogram = add_noise(spectrogram).astype(np.float32)
#     spectrogram = amplitude_scaling(spectrogram).astype(np.float32)
#     return spectrogram, notes

# def tf_augment_data(spectrogram, notes):
#     spectrogram, notes = tf.numpy_function(augment_data, [spectrogram, notes], [tf.float32, tf.int64])
#     spectrogram.set_shape([2048])  # Explicitly set the shape
#     notes.set_shape([49])          # Explicitly set the shape
#     return spectrogram, notes

# # Directory containing TFRecord files
# tfrecord_dir = 'GeneratedData'

# # Get list of all TFRecord files
# tfrecord_files = [os.path.join(tfrecord_dir, f) for f in os.listdir(tfrecord_dir) if f.endswith('.tfrecord')]

# # Create a dataset from the TFRecord files
# raw_dataset = tf.data.TFRecordDataset(tfrecord_files)

# # Parse the dataset
# parsed_dataset = raw_dataset.map(_parse_function)

# # Augment the dataset
# augmented_dataset = parsed_dataset.map(tf_augment_data)

# # Shuffle, batch, and prefetch the dataset
# batch_size = 32
# dataset = augmented_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2048, 1)),

    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(49, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# history = model.fit(dataset, epochs=50, validation_data=dataset, validation_steps=80, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

# model_save_path = '*saved_tf_models/BasicDenseGuitarNotePredictor.keras'

# # Save the model
# model.save(model_save_path)