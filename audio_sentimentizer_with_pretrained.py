#https://zenodo.org/records/1188976
from transformers import WhisperFeatureExtractor
#https://huggingface.co/learn/audio-course/chapter1/preprocessing
from datasets import Dataset
from pathlib import Path
import os
import pandas as pd
import librosa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import keras
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import librosa.display
import librosa.display as dsp
from sklearn.model_selection import train_test_split
from datasets import Audio
from datasets import load_dataset
  #Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
  #Vocal channel (01 = speech, 02 = song).
  #Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
  #Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
  #Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
  #Repetition (01 = 1st repetition, 02 = 2nd repetition).
  #Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

def get_subdirectories(directory_path):
    subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    return subdirectories

def get_labels(emotions):
    index = 0
    labels = {}
    gender=''
    for i in range(1, 3):
        #Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
        if i%2==0:
            gender='male'
        else:
            gender='female'

        for e in range(1, len(emotions)+1):
            emo = emotions[e]
            label = gender + "_" + emo
            labels[label] = index
            index+=1
    return labels

def load_data():
    directory_path = 'Audio_Speech_Actors'
    subdirectories = get_subdirectories(directory_path)
    x_data, y_data = get_data(subdirectories, directory_path)

    y_data = y_data['Label']

        # Extract features from the nested DataFrame
    features_list = [np.squeeze(feature) for feature in x_data['feature'].values]

            # Convert the list of features to a NumPy array
    x_data_array = np.array(features_list)
        #Splitting Dataset
    x_train, x_test, y_train, y_test = train_test_split(x_data_array, y_data, test_size=0.2, random_state=42)
        
        #Shuffling Dataset
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)
        
        #Splitting Test Dataset to have a Validation dataset.        
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=validation_size, random_state=42)

        #Convert the NumPy array to a TensorFlow tensor
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)

        #Printing Len of Datasets
    print("TRAIN Dataset Length:",len(x_train))
    print("TEST Dataset Length:",len(x_test))
    print("VAL Dataset Length:",len(x_val))


def assign_label(gender, emo, labels):
    label_str = gender + "_" + emo
    label = labels[label_str]
    return label


def parse_audio( full_file_path):
    sr=16000
    raw_audio, sr_original = librosa.load(full_file_path, res_type='kaiser_fast', duration=3, offset=0.5,sr=16000)

#    sr_original = np.array(sr_original)
#        # Find the indices of non-zero elements
#    nonzero_indices = np.nonzero(raw_audio)[0]                    
#        #Preprocessing to get to the first parts of data that contain values
#        # Get the index of the first non-zero element
#    if len(nonzero_indices) > 0:
#        first_nonzero_index = nonzero_indices[0]
#        raw_audio = raw_audio[first_nonzero_index:]
#    else:
#        print("The array contains only zeros.")
##        self.view_audio(X,sample_rate)
    #mfccs=np.mean(librosa.feature.mfcc(y=X, n_mfcc=13,),axis=0)
    return raw_audio

def get_data(subdirectories, directory_path):
    labels = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fearful', 7:'disgust', 8:'surprised'}
    emotions = get_labels(labels)
    max_length = 250
    path = Path(directory_path)
    file_names = []
    data_list = []
    x_data = pd.DataFrame(columns=['audio'])
    row = np.zeros(260)
    i=0
    for dir in subdirectories:
        for filename in os.listdir(os.path.join(directory_path, dir)):
            full_file_path = os.path.join(directory_path,dir, filename)
            if os.path.isfile(full_file_path):
                row = np.zeros(max_length)
                mfccs = parse_audio(full_file_path)
                len_row= len(mfccs)
                    # If the length of mfccs is greater than max length, truncate it

                if len_row > max_length:
                    mfccs = mfccs[:max_length]
                row[:len_row] = mfccs
                feature = row
                #x_data.loc[i]=[-(feature/100)]
                x_data.loc[i]=[feature]

                i+=1
                file_props = filename.split("-")
                modality = file_props[0]
                vocal_channel = file_props[1]
                emotion = int(file_props[2])
                emotional_intensity = file_props[3]
                statement = file_props[4]
                repetition = file_props[5]
                actor = file_props[6]
                #Removing Extension
                actor = actor.replace('.wav',"")
                if int(actor)%2==0:
                    gender='male'
                else:
                    gender='female'
                label = assign_label(gender, labels[emotion], emotions)
                # Create a dictionary with the extracted data
                file_data = {
                        #'Modality': modality,
                        #'VocalChannel': vocal_channel,
                        #'Emotion': emotion,
                        #'EmotionalIntensity': emotional_intensity,
                        #'Statement': statement,
                        #'Repetition': repetition,
                        #'Actor': actor,
                    'Label':label
                }
                
                    # Append the dictionary to the list
                data_list.append(file_data)

                    #BREAKING FOR QUICK TESTING PURPOSES NEED TO REMOVE LATER
                
                break
        # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data_list)
    return x_data, df






def prepare_dataset(example):
    audio = example["audio"]
    features = feature_extractor(
        audio["array"], sampling_rate=16000, padding=True
    )
    return features


feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")



def prepare_dataset(example):
    audio = example["audio"]
    features = feature_extractor(
        audio, sampling_rate=16000, padding=True
    )
    return features

def main():
        # Replace 'your_directory' with the path to your audio files
    directory_path = 'Audio_Song_Actors/Actor_01'

    # Define the dataset configuration
    dataset_name = 'your_dataset_name'
    # Define the dataset configuration
    dataset_config = f'audio/{directory_path}'

        # List all .wav files in the directory
    wav_files = [file for file in os.listdir(directory_path) if file.endswith('.wav')]

    # Load audio files and create a Dataset
    data = {'audio': [], 'path': []}

    for wav_file in wav_files:
        file_path = os.path.join(directory_path, wav_file)
        audio_data, sampling_rate = librosa.load(file_path, sr=None)
    
        data['audio'].append(audio_data)
        data['path'].append(file_path)

    # Convert the data dictionary to a Dataset
    dataset = Dataset.from_dict(data)

    # Print the dataset information
    print(dataset)
    #dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    dataset = dataset.map(prepare_dataset)
    print(dataset)


    example = dataset[0]
    input_features = example["input_features"]

    plt.figure().set_figwidth(12)
    librosa.display.specshow(
        np.asarray(input_features[0]),
        x_axis="time",
        y_axis="mel",
        sr=feature_extractor.sampling_rate,
        hop_length=feature_extractor.hop_length,
    )
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main() 