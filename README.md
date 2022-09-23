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
- Download the AI6G VM [here FILL THE CORRECT LINK]()
- [Install the Virtualbox software](https://www.virtualbox.org/wiki/Downloads)
- Open the installed Virtualbox software, click on the `file` menu and after `Import appliance` and choose the AI6G VM you downloaded on the previous step.

![image](https://user-images.githubusercontent.com/12988541/191989757-987f685f-42ff-4edd-a883-b9623e617297.png)

- After the VM was imported, the AI6G VM will appear in the virtualbox interface and you just need to start it clicking on Start button.
- Wait for 10 seconds (time to initialize the VM), open your browser in your host machine (not the VM) and access the link [localhost:4321/](http://localhost:4321/) that you will be able to access the Jupyter notebook server with the notebooks running into the AI6G VM.

#### 2) At Colab Google computers
- The Colab versions of the notebooks 02, 03, 04, 06, and 10 need to access data sets that are not on the repository, therefore it is necessary to follow these two steps:

1) Logged in your Google Account, enter in the link of the public folder storing the data set, named "ai6g_files" https://drive.google.com/drive/folders/1HcXb_fN590f7fesLQ1LLRoQCSuzCFZ9r 

2) After that, on you "Shared with me" (Compartilhados comigo) tab, go to the "ai6g_files" folder, use the right-click button, and press the button "Add shortcut to Google Drive" (Adicionar atalho ao Google Drive)

All set.

While executing the notebooks that need access to this data, the Colab will ask for you to login so that it can find the files.

#### 3) Using Conda to create an environment

git clone 

To create an environment and install dependencies on your computer:

To create the ai6g_env environment using conda:
```
conda create --name ai6g_env python=3.9.7
conda activate ai6g_env
```
(when using Jupyter notebooks, do not forget to choose this environment)

