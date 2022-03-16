import os
import glob
import pickle
import numpy as np
from vae import VAE
import librosa
import torch
import soundfile as sf
from emodb_preprocess import MinMaxNormalizer


HOP_LENGTH = 313
SAVE_DIR_ORIGINAL = "./samples/original"
SAVE_DIR_GENERATED = "./samples/generated"
MIM_MAX_VALUES_PATH = "./data/min_max_values.pkl"
SPECTROGRAMS_PATH = './data/spectrograms/'


def save_signals(signals, save_dir, sample_rate=16000):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)

def load_emodb(spectrogram_path):
    x_train = []
    file_paths = []
    i = 0
    for root, _, file_names in os.walk(spectrogram_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) 
            spectrogram = spectrogram[np.newaxis, ...]# (1, n_bins, n_frames)
            x_train.append(spectrogram)
            file_paths.append(file_path)
            i += 1
            if i > 10:
                break
    x_train = np.array(x_train)
    # x_train = x_train[np.newaxis, ...] # -> (1, #specs, 256, 256)
    print('spec shape: ', x_train.shape)
    return x_train, file_paths


def select_spectrograms(spectrograms, file_paths, min_max_values, num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrograms = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in file_paths]
    
    return sampled_spectrograms, sampled_min_max_values


class SoundGenerator:
  """
  SoundGenerator is responsible for genenrating audio from
  spectrogrmas
  """
  def __init__(self, vae, hop_length):
    self.vae = vae
    self.hop_length = hop_length
    self._min_max_normalizer = MinMaxNormalizer(0, 1)


  def generate(self, spectrograms, min_max_values):
    print(spectrograms.shape)
    spectrograms = torch.from_numpy(spectrograms)
    latent_representations, z_mean, z_log_var, generated_spectrograms, z = self.vae(spectrograms)
    signals = self.convert_spectrograms_to_audio(generated_spectrograms.cpu().detach().numpy(), min_max_values)
    return signals, latent_representations

  def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
    signals = []
    for spectrogram, min_max_value in zip(spectrograms, min_max_values):
      # reshape the log spectrogram
      log_spectrogram = spectrogram[0, :, :]
      # apply de-normalization
      denorm_log_spec = self._min_max_normalizer.denormalize(log_spectrogram, min_max_value["min"], min_max_value["max"])
      # log spectrogram -> spectrogram
      spec = librosa.db_to_amplitude(denorm_log_spec)
      # apply Griffin-Lim
      signal = librosa.istft(spec, hop_length=self.hop_length)
      # append signal to "signals"
      signals.append(signal)

    return signals



if __name__ == "__main__":
    # initialize sound generator
    model = VAE(256)
    models_path = f'./saved_models_256/'
    # model_names = glob.glob(models_path + '*.pth')
    # model_indices = [int(num.split('/')[-1].split('_')[1]) for num in model_names]
    # current_epoch = max(model_indices)
    # model_name = model_names[np.argmax(np.asarray(model_indices))]
    model_name = f'./saved_models_256/model_280_1510.0594750888424.pth'
    # Load state_dict
    model.load_state_dict(torch.load(model_name))
    ##################################################
    sound_generator = SoundGenerator(model, HOP_LENGTH)
    # load spectrograms + min max values
    with open(MIM_MAX_VALUES_PATH, "rb") as f:
      min_max_values = pickle.load(f)
      
    specs, file_paths = load_emodb(SPECTROGRAMS_PATH)
    # sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs, file_paths, min_max_values, 5)
    # generate audio for sampled spectrograms
    signals, latent_rep = sound_generator.generate(sampled_specs, sampled_min_max_values)
    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(sampled_specs, sampled_min_max_values)
    
    # save audio signals
    save_signals(original_signals, SAVE_DIR_ORIGINAL)
    save_signals(signals, SAVE_DIR_GENERATED)