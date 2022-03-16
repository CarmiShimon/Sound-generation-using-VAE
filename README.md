# Sound-generation-using-VAE
Pytorch implementation of Emotions generation with VAE using EmoV-DB.

- The idea behind this project is to build a machine learning model that could generate more samples of voiced emotions.
- Using the pre-trained model you could use both the latent vector of your voice for classification and for generation a new sample which sounds similar to your voice by using the reparametrization trick. 

**Dataset:**
[EmoV-DB](https://github.com/numediart/EmoV-DB)

### Audio files:
**Waveform**
![Audio](./images/audio.png)
**Spectrogram**
![Spectrogram](./images/spectrogram.png)

### pre-trained models
[256X256 spectrogram model](https://drive.google.com/file/d/1B6yFE6gwGfqQrfuOapahrpdfNig4QQUj/view?usp=sharing)
Place it under 'saved_models_256'
