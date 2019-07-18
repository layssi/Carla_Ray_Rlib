# Carla-Ray-RLib
Reinforcement Learning with Rlib and Carla

![Cloud Based Autonomous Driving RL](/home/salstouhi/Desktop/Carla_Ray_Rlib/docs/thumbnail_CARLA_RAY.jpg  "Cloud Based Autonomous Driving RL")


# Setup Carla
## Download Carla Binaries
### (Option 1) Release 0.9.6:
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz  
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Town06_0.9.6.tar.gz  
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Town07_0.9.6.tar.gz  

### (Option 2)Latest Build (Warning. Not Tested)
http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Dev/CARLA_Latest.tar.gz

## Setup Carla
### To lower Carla GPU usage:
Under your Carla/CARLAUE4/Config, edit *DefaultEngine.ini*:  
**r.TextureStreaming=True**

### (Optional) Mouse Unlock
To prevent the freezing of the mouse (which is very annoying).Under your Carla/CARLAUE4/Config, edit *"DefaultInput.ini"*:  
**bCaptureMouseOnLaunch=False**  
**DefaultViewportMouseCaptureMode=CaptureDuringMouseDown**  
**DefaultVi== ewportMouseLockMode=DoNotLock**  

# Setup Environment
## Install Environment
Download and install conda: https://www.anaconda.com/distribution/  
conda env create --name [**enviroment-name**] -f=requirements.txt   
conda activate [**enviroment-name**]  
Conda might ask you to do "conda init bash"  
conda install -c anaconda tensorflow-gpu  

## Update Carla path
In "helper" folder, change the location of your Carla path in file "CARLA_PATH.txt".  Example: ~/home/Carla_Simulator
## (optional) Pycharm setup
Start a pycharm project and choose the anaconda envirometn create [**enviroment-name**] 


