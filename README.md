# Speech2RIR

This is the official implementation of reverberant speech to room impulse response estimator. We trained our network using Room Impulse Responses from [**SoundSpaces-NVAS dataset**](https://ieeexplore.ieee.org/document/10204911) and clean speech from [**LibriSpeech dataset**](https://ieeexplore.ieee.org/document/7178964) 



# Requirements

```
Python 3.8+
Cuda 11.0+
PyTorch 1.10+
numpy
pygsound
wavefile
tqdm
scipy
soundfile
librosa
cupy-cuda11x
torch_stoi
tensorboardX
pyyaml
sudo apt-get install p7zip-full  
```

# RIR and Clean Speech Dataset
Run the following script to download SoundSpaces-NVAS and LibriSpeech Dataset
```
bash download_data.sh
```
# Reverberant Speech Augmentation

Run the following commands to augment **Reverberant Speech** to train and test.

```
./batch_flac2wav.sh data/LibriSpeech-wav
 python3 pickle_generator.py

```

# Download Trained Model
To download our trained with checkpoint at **1,000,000** Run the following command

```
source download_model.sh
```

# Testing
To test the trained model run the following command

```
bash submit_autoencoder.sh --start 2
```

# Training
To train our network, run the following command

```
bash submit_autoencoder.sh --start 0 --stop 0 --tag_name "autoencoder/symAD_vctk_48000_hop300"
```
To resume training on a saved model at a particular step (e.g., 1,000,000 steps) run the following command

```
bash submit_autoencoder.sh --start 1 --stop 1 --resumepoint 1000000 --tag_name "autoencoder/symAD_vctk_48000_hop300"
```

# Citations

Our model is built using the architectures from [**S2IR**](https://ieeexplore.ieee.org/abstract/document/10094770/citations?tabFilter=papers#citations) and [**AV-RIR**](https://openaccess.thecvf.com/content/CVPR2024/html/Ratnarajah_AV-RIR_Audio-Visual_Room_Impulse_Response_Estimation_CVPR_2024_paper.html). If you use our **Speech2RIR**, please consider citing

```
@INPROCEEDINGS{10094770,
  author={Ratnarajah, Anton and Ananthabhotla, Ishwarya and Ithapu, Vamsi Krishna and Hoffmann, Pablo and Manocha, Dinesh and Calamia, Paul},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Towards Improved Room Impulse Response Estimation for Speech Recognition}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Measurement;Error analysis;Estimation;Signal processing;Benchmark testing;Generative adversarial networks;Acoustics;room impulse response;blind estimation},
  doi={10.1109/ICASSP49357.2023.10094770}}
```

```
@InProceedings{Ratnarajah_2024_CVPR,
    author    = {Ratnarajah, Anton and Ghosh, Sreyan and Kumar, Sonal and Chiniya, Purva and Manocha, Dinesh},
    title     = {AV-RIR: Audio-Visual Room Impulse Response Estimation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {27164-27175}
}
```






If you use [**SoundSpaces-NVAS dataset**](https://ieeexplore.ieee.org/document/10204911), please consider citing

```
@INPROCEEDINGS{10204911,
  author={Chen, Changan and Richard, Alexander and Shapovalov, Roman and Ithapu, Vamsi Krishna and Neverova, Natalia and Grauman, Kristen and Vedaldi, Andrea},
  booktitle={2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Novel-View Acoustic Synthesis}, 
  year={2023},
  volume={},
  number={},
  pages={6409-6419},
  keywords={Location awareness;Visualization;Computational modeling;Transforms;Benchmark testing;Rendering (computer graphics);Pattern recognition;Multi-modal learning},
  doi={10.1109/CVPR52729.2023.00620}}
```






If you use [**LibriSpeech dataset**](https://ieeexplore.ieee.org/document/7178964). please consider citing

```
@INPROCEEDINGS{7178964,
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Librispeech: An ASR corpus based on public domain audio books}, 
  year={2015},
  volume={},
  number={},
  pages={5206-5210},
  keywords={Resource description framework;Genomics;Bioinformatics;Blogs;Information services;Electronic publishing;Speech Recognition;Corpus;LibriVox},
  doi={10.1109/ICASSP.2015.7178964}}
```


