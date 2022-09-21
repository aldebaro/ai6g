# ai6g
Artificial intelligence / machine learning in 5G / 6G mobile networks

## Update requirements for conda or pip users

1) Update your conda version 
```
conda update -n base -c defaults conda
```
2) Install pip in your conda env
```
conda install pip
```
3) Install the marvelous pipreqs:
```
pip install pipreqs
```
4) Generate a requirements.txt file only with the modules you are effectively invoking:
```
pipreqs d:\github\ai6g
```
where d:\github\ai6g is your folder with the notebooks.

5) Install the required modules
```
pip install -r requirements.txt
```

#### There are alternatives though:
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
pip install tensorflow==2.9.2
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
Note that you can eventually use the latest Tensorflow version (instead of the version 2.9.2 imposed above) using:
```
pip install tensorflow
```
but this code was tested with TF 2.9.2.

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

01_detection_qam_over_awgn.ipynb – Detection of QAM symbols transmitted over AWGN. The problem is posed as classification and the machine learning solutions are compared to the theoretical optimum (minimum) error rate.

02_channel_estimation_digital_mimo.ipynb - Downlink channel estimation using narrowband synthetic channel data and assuming the receiver quantizes the signals with 1-bit ADCs

03_channel_estimation_hybrid_mimo.ipynb - Channel estimation using ITU challenge problem organized by NCSU and using Raymobtime. Illustrates Keras support to complex numbers

04_beam_selection_analog_mimo.ipynb - Beam selection in MIMO using LIDAR using the Raymobtime channels

05_federated_learning_beam_selection.ipynb - An example of Federated Learning applied to beam selection using LIDAR data - From ITU Challenge - https://github.com/ITU-AI-ML-in-5G-Challenge/PS-012-ML5G-PHY-Beam-Selection_Imperial_IPC1

06_channel_compression_auto_encoder.ipynb -  CSI (channel) compression with CSInet (auto-encoder / end-to-end / encoding) based on code from https://github.com/sydney222/Python_CsiNet

07_end_to_end_auto_encoder.ipynb - Autoencoder using NVIDIA's Sionna

08_finite_mdp_reinforcement_learning.ipynb - Finite-stete RL toy problem

09_deep_reinforcement_learning.ipynb - DRL-based Resource allocation

10_channel_estimation_using_gan.ipynb - GAN code based on https://github.com/YudiDong/Channel_Estimation_cGAN

11_time_series_processing.ipynb  – Synthetic channel data. Sequence processing with LSTM
