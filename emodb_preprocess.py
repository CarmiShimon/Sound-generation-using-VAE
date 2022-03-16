"""
"""
import os
import librosa
from scipy.io.wavfile import write
import numpy as np
import tqdm
import pickle
import sys
import glob
import argparse
import random
random.seed(10)

class Loader:
    """Loader is responsible for loading an audio file."""
    
    def __init__(self, sr, duration, mono):
        self.sr = sr
        self.duration = duration
        self.mono = mono
        
    def load(self, file_path):
        signal = librosa.load(file_path, sr=self.sr, mono=self.mono)[0]
        return signal
        
        
class Padder:
    """Padder is responsible to apply padding to an array."""
    
    def __init__(self, mode="constant"):
        self.mode = mode
        
    def pad(self, signal, padded_samples):
        signal = librosa.util.pad_center(signal, padded_samples, mode='constant')
        return signal
    
    
class SilenceRemoval:
    """"SilenceRemoval is responsible for removing leading and trailing silence."""
    
    def __init__(self, top_db=25):
        self.top_db = top_db
        
    def trim(self, audio, expected_length):
        # clean_audio, _ = librosa.effects.trim(audio, top_db=self.top_db)
        ignore_samples = (len(audio) - expected_length)//2
        trimmed_audio = audio[ignore_samples:-(ignore_samples+1)]
        if len(trimmed_audio) < expected_length:
            return audio[ignore_samples:-(ignore_samples)]
        return audio[ignore_samples:-(ignore_samples+1)]
        
        
class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a 
    time-series signal.
    """
    
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract(self, signal):
        stft = librosa.stft(signal, n_fft=self.frame_size, 
                            hop_length=self.hop_length)[:-1] # (1+frame_size / 2, num_frames) 
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram
        

class MinMaxNormalizer:
    """MinMaxNormalizer applies min max normalization to an array."""
    
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
    
    def normalize(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        # -> [0, 1]
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array
    
    def denormalize(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class Saver:
    """saver is responsible to save features, and the min max values"""
    
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir
        os.makedirs(f'{feature_save_dir}train', exist_ok=True)
        os.makedirs(f'{feature_save_dir}test', exist_ok=True)
        os.makedirs(f'{feature_save_dir}eval', exist_ok=True)
        
    
    def save_feature(self, feature, file_path, save_dir):
        save_path, dir_name = self._generate_save_path(file_path, save_dir) # save dir is train/test/eval
        os.makedirs(os.path.join(self.feature_save_dir, dir_name), exist_ok=True)
        np.save(save_path, feature)
        return save_path
    
    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)
    
    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
    
    def _generate_save_path(self, file_path, save_dir):
        dir_name = f'{save_dir}/{file_path.split("/")[-2]}'
        file_name = os.path.join(dir_name, os.path.split(file_path)[1][:-4])
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path, dir_name
        

class TrainTestSplit:
    """
    TrainTestSplit is responsible for splitting wav files to train, test, eval
    Also, it makes sure the train set is balanced across all classes
    """
    def __init__(self, train_prc):
        self.train_prc = train_prc
        self.test_prc = (1 - self.train_prc)//2
        self.eval_prc = self.test_prc
    
    def split(self, audio_dir):
        audio_files = glob.glob(audio_dir + '*/*.wav')
        files_dict = {'sleepy': [], 'angry': [], 'disgust': [], 'amused': [], 'neutral': []}
        
        for file in audio_files:
            if 'sleepiness' in file:
                files_dict['sleepy'].append(file)
            elif 'anger' in file:
                files_dict['angry'].append(file)
            elif 'disgust' in file:
                files_dict['disgust'].append(file)
            elif 'amused' in file:
                files_dict['amused'].append(file)
            elif 'neutral' in file:
                files_dict['neutral'].append(file)


        min_class_num_files = len(min(files_dict.values(), key=len))
        # min_list_files = [key for key, value in files_dict.items() if value == min_value]
        print('train files in each class = ', min_class_num_files)
        # l = np.arange(min_value)
        # np.random.shuffle(l)
        train_files = []
        test_files = []
        eval_files = []
        for files in files_dict.values():
            random.shuffle(files)
            train_files += files[:int(self.train_prc*min_class_num_files)]
            other_files = files[int(self.train_prc*min_class_num_files):]
            test_files += other_files[:int(0.5*len(other_files))]
            eval_files += files[int(0.5*len(other_files)):]
        
        return train_files, test_files, eval_files
        
        

class PreprocessingPipeline:
    """PreprocessingPipeline preprocess audio files in a directory,
        applying the following steps to each file:
            1 - load a file
            2 - pad or truncate if necessary
            3 - extracting log spectrogram from signal
            4 - normalize spectrogram
            5 - save the normalized spectrogram
    Storing the min max values for all the log spectrogrmas.
    """
    
    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.saver = None
        self.trimmer = None
        self.splitter = None
        self.min_max_values = {}
        self._num_expected_samples = None
        self._loader = None
        
    
    @property
    def loader(self):
        return self._loader
        
        
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sr * loader.duration)
        
        
    def process(self, audio_files_dir):
        train_files, test_files, eval_files = self.splitter.split(audio_files_dir)
        print('total train files: ', len(train_files))
        print('total test files: ', len(test_files))
        print('total eval files: ', len(eval_files))
        for root, sub_dirs, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(file_path)
                if file_path in train_files:
                    save_dir = 'train'
                elif file_path in test_files:
                    save_dir = 'test'
                else:
                    save_dir = 'eval'
                self._process_file(file_path, save_dir)
                print(f"Processed file {file_path}")
                
        self.saver.save_min_max_values(self.min_max_values)
        
    
    def _process_file(self, file_path, save_dir):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        else:
            signal = self.trimmer.trim(signal, self._num_expected_samples)
        feature = self.extractor.extract(signal)
        print('shape: ', feature.shape)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path, save_dir)
        print('saved_path', save_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())
        
    def _is_padding_necessary(self, signal):
        if len(signal) < self._num_expected_samples:
            return True
        return False
        
    def _apply_padding(self, signal):
        padded_signal = self.padder.pad(signal, self._num_expected_samples)
        return padded_signal
        
    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_size', default=512, type=int)
    parser.add_argument('--hop_length', default=256, type=int)
    parser.add_argument('--duration', default=5, type=int)
    parser.add_argument('--sr', default=16000, type=int)
    
    args = parser.parse_args()
    
    FRAME_SIZE = args.frame_size
    HOP_LENGTH = args.hop_length
    DURATION = args.duration
    SR = args.sr
    MONO = True
    
    local_dir = os.getcwd()
    SPECTROGRAMS_SAVE_DIR = "./data/spectrograms/"
    MIM_MAX_VALUES_PATH = "./data/"
    FILES_DIR = './data/audio/'
    
    # instantiate all objects
    loader = Loader(SR, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normalizer = MinMaxNormalizer(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIM_MAX_VALUES_PATH)
    trimmer = SilenceRemoval()
    splitter = TrainTestSplit(0.8)
    
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normalizer = min_max_normalizer
    preprocessing_pipeline.trimmer = trimmer
    preprocessing_pipeline.saver = saver
    preprocessing_pipeline.splitter = splitter
    
    preprocessing_pipeline.process(FILES_DIR)