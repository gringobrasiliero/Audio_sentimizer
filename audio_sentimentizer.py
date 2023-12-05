#https://zenodo.org/records/1188976

#https://github.com/MiteshPuthran/Speech-Emotion-Analyzer


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
  #Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
  #Vocal channel (01 = speech, 02 = song).
  #Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
  #Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
  #Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
  #Repetition (01 = 1st repetition, 02 = 2nd repetition).
  #Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

class Audio_sentimizer():

    def __init__(self):
        self.model_name = 'audio_sentimizer'
        self.max_length = 250
        self.emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

        #Train, Test, and Validation datasets loaded in 'load_data' function
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        

        #Model - Defined in Load_Model()
        self.model = None

        #HYPERPARAMETERS
        self.epochs = 1
        self.batch_size = 64
        self.dropout = 0.2
        self.validation_size = 0.5 # Size of Validation data set - Split from test data set.


        #Callbacks
        self.es_callback = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', patience=3)

        pass


    def get_subdirectories(self, directory_path):
        subdirectories = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
        return subdirectories

    def parse_audio(self, full_file_path):
        X, sample_rate = librosa.load(full_file_path, res_type='kaiser_fast', duration=3, offset=0.5,sr=22050*2)
        sample_rate = np.array(sample_rate)
        # Find the indices of non-zero elements
        nonzero_indices = np.nonzero(X)[0]                    
        #Preprocessing to get to the first parts of data that contain values
        # Get the index of the first non-zero element
        if len(nonzero_indices) > 0:
            first_nonzero_index = nonzero_indices[0]
            print("First index of non-zero element:", first_nonzero_index)
            print(X[0])
            X = X[first_nonzero_index:]
            print(X[0])
        else:
            print("The array contains only zeros.")
            self.view_audio(X,sample_rate)
        mfccs=np.mean(librosa.feature.mfcc(y=X, n_mfcc=13,),axis=0)
        return mfccs

    def evaluate_test_data(self):
        eval_results = self.model.evaluate(self.x_test,self.y_test,batch_size=self.batch_size)
        loss, accuracy = eval_results
        print("Test Loss:",loss)
        print("Test Accuracy:",accuracy)
        pass


    def get_data(self, subdirectories, directory_path):
        path = Path(directory_path)
        file_names = []
        data_list = []
        x_data = pd.DataFrame(columns=['feature'])
        row = np.zeros(260)
        i=0
        for dir in subdirectories:
            for filename in os.listdir(os.path.join(directory_path, dir)):
                print(filename)
                full_file_path = os.path.join(directory_path,dir, filename)
                print(full_file_path)
                if os.path.isfile(full_file_path):
                    row = np.zeros(self.max_length)
                    mfccs = self.parse_audio(full_file_path)
                    len_row= len(mfccs)
                    # If the length of mfccs is greater than max length, truncate it

                    if len_row > self.max_length:
                        mfccs = mfccs[:self.max_length]
                    row[:len_row] = mfccs
                    feature = row
                    x_data.loc[i]=[-(feature/100)]

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
                
                                    # Create a dictionary with the extracted data
                    #file_data = {
                    #    'Modality': modality,
                    #    'VocalChannel': vocal_channel,
                    #    'Emotion': emotion,
                    #    'EmotionalIntensity': emotional_intensity,
                    #    'Statement': statement,
                    #    'Repetition': repetition,
                    #    'Actor': actor
                    #}
                    file_data = {
                        'Modality': modality,
                        'VocalChannel': vocal_channel,
                        'Emotion': emotion,
                        'EmotionalIntensity': emotional_intensity,
                        'Statement': statement,
                        'Repetition': repetition,
                        'Actor': actor
                    }
                
                    # Append the dictionary to the list
                    data_list.append(file_data)
                    break
                break
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data_list)
    
        return x_data, df
    
    
    def view_audio(self, audio, sample_rate):
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio, sr=sample_rate)
        # Add labels and title
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Waveplot of Audio Signal')

        # Show the plot
        plt.show()


        d = librosa.stft(audio)
        D = librosa.amplitude_to_db(np.abs(d),ref=np.max)
        fig,ax = plt.subplots(2,1,sharex=True,figsize=(10,10))
        img = dsp.specshow(D, y_axis='linear', x_axis='s',sr=sample_rate,ax=ax[0])
        ax[0].set(title='Linear frequency power spectrogram')
        ax[0].label_outer()
        dsp.specshow(D,y_axis='log',x_axis='s',sr=sample_rate,ax=ax[1])
        ax[1].set(title='Log frequency power spectrogram')
        ax[1].label_outer()
        fig.colorbar(img, ax=ax, format='%+2.f dB')
        plt.show()


    def load_model(self):
        exists = os.path.isdir(self.model_name)
        model = None
        if exists:
            model = tf.keras.models.load_model(self.model_name)
        return model
    
    def define_model(self):
        es_callback = keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, input_shape=(250,), activation='relu'), #Original Height, Original Width, and 400 Pixels
            tf.keras.layers.Dropout(0.2), #20% dropout rate
            tf.keras.layers.Dense(58, activation=tf.nn.softmax) #57 Different Categories = 58
            ])
        
        self.model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        pass

    
    def train_model(self):
        #Stops training early if Accuracy is decreasing
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.x_val, self.y_val), callbacks=[self.es_callback])    
        self.model.save(self.model_name)
        return history

    def make_predictions(self, num_predictions):
        predictions = self.model.predict(self.x_test)

        for i in range(num_predictions):
            predicted=predictions[i].argmax()
            label_string = "Predicted:" + self.emotions[predicted] + "\nActual:" + self.emotions[self.y_test[i][0]] + "\n\n"
        
    def load_data(self):
        directory_path = 'Audio_Speech_Actors'
        subdirectories = self.get_subdirectories(directory_path)
        x_data, y_data = self.get_data(subdirectories, directory_path)

        y_data = y_data['Emotion']
        print(x_data.iloc[0][0])
            #x_data = x_data.values
        print(x_data)
            # Extract features from the nested DataFrame
        features_list = [np.squeeze(feature) for feature in x_data['feature'].values]

            # Convert the list of features to a NumPy array
        x_data_array = np.array(features_list)
        #Splitting Dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_data_array, y_data, test_size=0.2, random_state=42)
        
        #Shuffling Dataset
        self.x_train, self.y_train = shuffle(self.x_train, self.y_train, random_state=42)
        self.x_test, self.y_test = shuffle(self.x_test, self.y_test, random_state=42)
        
        #Splitting Test Dataset to have a Validation dataset.        
        self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(self.x_test, self.y_test, test_size=self.validation_size, random_state=42)

        #Convert the NumPy array to a TensorFlow tensor
        self.x_train = tf.convert_to_tensor(self.x_train, dtype=tf.float32)
        self.x_test = tf.convert_to_tensor(self.x_test, dtype=tf.float32)
        self.x_val = tf.convert_to_tensor(self.x_val, dtype=tf.float32)

        
        

            #y_data = tf.convert_to_tensor(y_data, dtype=tf.int32) NOT NEEDED?


    def display_model_history(self, history):
        #Display Accuracy History Plot
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig('training_history.png')
        plt.waitforbuttonpress()
        plt.close()
        #Display Loss History Plot
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig('training_loss.png')
        plt.waitforbuttonpress()
        plt.close()
        pass
    




def main():
    x = Audio_sentimizer()
    x.load_data()
    # Replace 'your_directory_path' with the path to your target directory
    x.define_model()
    history = x.train_model()
    x.evaluate_test_data()
    x.make_predictions(5)
 

if __name__ == "__main__":
    main()