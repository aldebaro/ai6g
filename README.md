# ai6g
Software for the tutorial "Artificial intelligence / machine learning in 5G / 6G mobile networks" by Aldebaro Klautau at SBrT'2022 in Santa Rita, Brazil. Several people contributed and the notebooks give the credits.

## Conventions

Notebook name: The names are indicated below. Note that each notebook is associated to a second one its "solution". For instance, 01_detection_qam_over_awgn.ipynb has the associated solutions_01_detection_qam_over_awgn.ipynb

Folder name: All the files needed by notebook number x are in a folder with a name that starts with files_x_string. For instance, the Python and dataset files associated to notebook 01_detection_qam_over_awgn.ipynb are located in files_01_detection. We keep only the first string of the notebook name. All these folders will be located both at the local disk and, to be reached by Colab, at LASSE's google drive.

## Jupyter Notebooks

01_detection_qam_over_awgn.ipynb – Detection of QAM symbols transmitted over AWGN. The problem is posed as classification and the machine learning solutions are compared to the theoretical optimum (minimum) error rate.

02_channel_estimation_digital_mimo.ipynb - Downlink channel estimation using narrowband synthetic channel data and assuming the receiver quantizes the signals with 1-bit ADCs

03_channel_estimation_hybrid_mimo.ipynb - Channel estimation using ITU challenge problem organized by NCSU and using Raymobtime. Illustrates Keras support to complex numbers

04_beam_selection_analog_mimo.ipynb - Beam selection in MIMO using LIDAR using the Raymobtime channels

05_federated_learning_beam_selection.ipynb - An example of Federated Learning applied to beam selection using LIDAR data - From ITU Challenge - https://github.com/ITU-AI-ML-in-5G-Challenge/PS-012-ML5G-PHY-Beam-Selection_Imperial_IPC1

06_channel_compression_auto_encoder.ipynb -  CSI (channel) compression with CSInet (auto-encoder / end-to-end / encoding) based on code from https://github.com/sydney222/Python_CsiNet

07_end_to_end_auto_encoder.ipynb - Autoencoder using NVIDIA's Sionna - https://developer.nvidia.com/sionna

08_finite_mdp_reinforcement_learning.ipynb - RL toy problem using a finite-state Markov decision process (MDP) to illustrate Bellman equations and modeling of RL without neural networks.

09_deep_reinforcement_learning.ipynb - DRL-based resource allocation

10_channel_estimation_using_gan.ipynb - GAN code based on https://github.com/YudiDong/Channel_Estimation_cGAN

11_time_series_processing.ipynb  – Synthetic channel data. Sequence processing with LSTM for a QAM detection problem similar to the one in notebook 01_detection_qam_over_awgn.ipynb

## Installation

### The datasets (not the Python code) are available from

....zip at google drive

### The software organized as Jupyter notebooks

You have some options to execute the notebooks:

#### 1) From a VirtualBox virtual machine (VM)

- provide instructions

#### 2) At Colab Google computers

#### 3) Using Conda to create an environment

git clone 

To create an environment and install dependencies on your computer:

To create the ai6g_env environment using conda:
```
conda create --name ai6g_env python=3.9.7
conda activate ai6g_env
```
(when using Jupyter notebooks, do not forget to choose this environment)

