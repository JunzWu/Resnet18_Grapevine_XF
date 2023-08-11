# Resnet18_Grapevine_XF
## Dataset Downlaod:
1. Download the files in this link and put them in the **datasets** directory:

https://drive.google.com/drive/u/0/folders/1fnC7UmE8_nxLF1R7hKqInXNy4sHByLzC


## Installation
1. Install the **annconda**: https://docs.anaconda.com/free/anaconda/install/index.html
2. Create the environment:
   `conda create --name XF python=3.8`
4. Install the **pytorch** :
   
   GPU version: `conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch`
   
   CPU version: `conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cpuonly -c pytorch`
5. Install other libiraries:
   
   `pip install -r requirement.txt`
## Getting Started
1. Do the training:

   `python train.py`
2. Do the test:

   `python test.py`
3. Do the visualization of Saliency Map:
   
   `python visualize.py`
