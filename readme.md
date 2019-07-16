# Carla-Ray-RLib
Reinforcement Learning with Rlib and Carla

# Setup Carla
## Download Carla Binaries
### (Option 1)Latest Build (Binaries)
http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Dev/CARLA_Latest.tar.gz

### (Option 2) Latest Release:

## Setup Carla
### To lower Carla GPU usage:
####Under your Carla/CARLAUE4/Config
In "DefaultEngine.ini":
**r.TextureStreaming=True**

### (Optional) Mouse Unlock
To prevent the freezing of the mouse (which is very annoying)
In "DefaultInput.ini":
**bCaptureMouseOnLaunch=False**
**DefaultViewportMouseCaptureMode=CaptureDuringMouseDown**
**DefaultVi== ewportMouseLockMode=DoNotLock**

# Setup Environment
## Install Environment
Download and install conda: https://www.anaconda.com/distribution/

conda env create --name [**enviroment-name**] -f=conda_environment_export.yml
conda activate [**enviroment-name**] 
Conda might ask you to do "conda init bash"
conda install -c anaconda tensorflow-gpu

## Update Carla path
In "helper" folder, change the location of your Carla path in file "CARLA_PATH.txt".  Example: ~/home/Carla_Simulator
## (optional) Pycharm setup
Start a pycharm project and choose the anaconda envirometn create [**enviroment-name**] 


