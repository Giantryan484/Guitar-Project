import os
import shutil

# List of folders from which to delete files
# folders_to_clean = ["MIDIs", "WAVs", "Spectrograms", "GeneratedData", "Data(4BackContext)"]
folders_to_clean = ["tfrecords"]

def delete_files_in_folder(folder_path):
    """ Deletes all files in the specified folder without removing the folder itself """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Deleted directory {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Loop through the folders and clear them
for folder in folders_to_clean:
    delete_files_in_folder(folder)
