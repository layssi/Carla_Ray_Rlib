# Carla-Ray-RLib
Reinforcement Learning with Rlib and Carla

![Cloud Based Autonomous Driving RL](https://github.com/layssi/Carla_Ray_Rlib/blob/master/docs/thumbnail_CARLA_RAY.jpg  "Cloud Based Autonomous Driving RL")


# Setup Carla
## Download Carla Binaries
### (Option 1) Release 0.9.6:
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz  
**Optional:** wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Town06_0.9.6.tar.gz  
**Optional:** wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Town07_0.9.6.tar.gz  

### (Option 2)Latest Build (Warning. Not Tested)
http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/Dev/CARLA_Latest.tar.gz

## Setup Carla
### To lower Carla GPU usage (This might be already set):
Under your Carla/CARLAUE4/Config, edit *DefaultEngine.ini*:  
**r.TextureStreaming=True**

### (Optional) Mouse Unlock
To prevent the freezing of the mouse (which is very annoying).Under your Carla/CARLAUE4/Config, edit *"DefaultInput.ini"*:  
**bCaptureMouseOnLaunch=False**  
**DefaultViewportMouseCaptureMode=CaptureDuringMouseDown**  
**bDefaultViewportMouseLock=DoNotLock**  
**DefaultViewportMouseLockMode=DoNotLock**  

# Setup Environment
## Install Environment
Download and install conda: https://www.anaconda.com/distribution/  
conda env create --name [**enviroment-name**] -f=requirements.yml   
conda activate [**enviroment-name**]  
conda install -c anaconda tensorflow-gpu  

## Update Carla path
In "helper" folder, change the location of your Carla path in file "CARLA_PATH.txt".  Example: ~/home/Carla_Simulator
## (optional) Pycharm setup
Start a pycharm project and choose the anaconda envirometn create [**enviroment-name**] 



# Cloud Setup
There is an ami available on ec2 with everything setup and no display.  
Just find "ami-070f500a304414585" and start the machine.  
Run "source ~/.bashrc" and run "python3 carla_env.py" or "python3 vision_algorithm.py"  
