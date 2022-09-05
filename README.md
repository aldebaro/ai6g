# ai6g
Artificial intelligence / machine learning in 5G / 6G mobile networks

## Update requirements for conda or pip users

Using the marvelous pipreqs:
```
pip install pipreqs
pipreqs d:\github\ai6g
```
where d:\github\ai6g is your folder with the notebooks.

See https://stackoverflow.com/questions/64500342/creating-requirements-txt-in-pip-compatible-format-in-a-conda-virtual-environmen

Using conda and saving all installed modules:
```
conda env export > environment.yml --no-builds
```

## Installation

### Using Conda

To create the environment using conda:
```
conda create --name ai6g python=3.9.7
conda activate ai6g
```

To install TF 2 from 
https://www.tensorflow.org/install/pip#windows-native

with GPU
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install tensorflow
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Because I got the error message:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow-probability 0.17.0 requires decorator, which is not installed.
```
I have executed:
```
pip install decorator
```

I also installed stable-baseline using the instructions at
https://github.com/conda-forge/stable-baselines3-feedstock
In summary:
```
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install stable-baselines3
```

### Using ...

## Conventions

Please note the convention regarding naming and... follow it

Notebook name: The names are indicated below. Note that each notebook is associated to a second one its "solution". For instance, 01_detection_qam_over_awgn.ipynb has the associated solutions_01_detection_qam_over_awgn.ipynb

Folder name: All the files needed by notebook x are in a folder with a name that starts with files_x. For instance, the Python and dataset files associated to notebook 01_detection_qam_over_awgn.ipynb are located in files_01_detection. We keep only the first string of the notebook name. All these folders will be located both at the local disk and, to be reached by Colab, at LASSE's google drive.

## Jupyter Notebooks and corresponding responsible person(s)

01_detection_qam_over_awgn.ipynb – Classification with AWGN and artificial channels using several classifiers of scikit-learn, do not use Keras. We have the theoretical best result - Concepts: 1) increase number of sampling examples, 2) multi-condition training, with several SNRs to show generalization capability. The solution shows the sample complexity for several classifiers. Lesson: nothing better than more data - Douglas e José

02_channel_estimation_digital_mimo.ipynb - Downlink channel estimation using three datasets: a) synthetic and b) Raymobtime. Again, we do not discuss the channel much but can use narrowband from Raymobtime. The synthetic channel is obtained by different positions of UEs. The BS has a fixed position. Each "site" has 3 clusters of UEs. We distinguish the sites when we compose training and test sets, and illustrate different results when: a) the sets mix examples from all sites, b) when trained and tested with data from distinct sites. Concepts: regression, low-resolution, and site-dependent models, transfer learning - Yuichi (+ código do Wesin)

03_analog_mimo_beam_selection.ipynb - Beam selection in MIMO using LIDAR. We use the Raymobtime channels but do not discuss / describe the channels. Concepts: 1) how to design the input features, 2) LOS versus NLOS. We use the Sequential Keras API, not the functional API. We do not give examples of combining the features, but we teach how to change the parameters of the LIDAR-based system. We assume noiseless channel and matched sites (training and test with the same site). The solutions can provide the sample complexity for some options - João e Luan

04_channel_estimation_hybrid_mimo.ipynb - Downlink (or uplink) channel estimation using Nuria's ITU challenge (Raymobtime). Concept: Hybrid architecture, Keras support to complex numbers, importance of additional modeling (signal processing) - Muller, Claudio e Daniel

05_federated_learning_beam_selection.ipynb - An example of Federated Learning applied to beam selection using LIDAR data - From ITU Challenge - https://github.com/ITU-AI-ML-in-5G-Challenge/PS-012-ML5G-PHY-Beam-Selection_Imperial_IPC1 - Concepts: federated learning. Ailton

06_channel_compression_auto_encoder.ipynb -  Channel compression with CSInet (auto-encoder / end-to-end / encoding) - https://github.com/sydney222/Python_CsiNet - Concepts: CSI (channel) compression, auto-encoder- Damasceno e Davi

07_end_to_end_auto_encoder.ipynb - Autoencoder using NVIDIA's Sionna. Concept: end-to-end - Luan 

08_finite_mdp_reinforcement_learning.ipynb - RL paper SBrT- Concept: RL - Rebecca

09_deep_reinforcement_learning.ipynb - RL-based Resource allocation – Cleverson

10_channel_estimation_using_gan.ipynb - GAN: https://github.com/YudiDong/Channel_Estimation_cGAN - Yuichi (note: Wesin has executed this code and I know the author)

11_time_series_processing.ipynb  – Synthetic channel data. Sequences, LSTM + GRU + Transformer - codigo do Globecom for synthetic data, and tracking? - Rodrigo

12_log_files_anomaly_detection.ipynb - An example of Anomaly detection – From ITU Challenge? 
https://aiforgood.itu.int/event/anomaly-detection-based-on-log-analysis/
https://github.com/ITU-AI-ML-in-5G-Challenge/ITU-ML5G-PS-CN2.1-Network-cloud-equipment-anomaly-and-root-cause-analysis
but When I browse this anomaly detection site I do not find any code, but a .key file - João e Luan
