import urllib.request
import zipfile
import os

url = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
filename = "horse-or-human.zip"
training_dir = "horse-or-human/training"

os.makedirs(training_dir, exist_ok=True)

urllib.request.urlretrieve(url, filename)

with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall(training_dir)

print("Dataset downloaded and extracted successfully.")
