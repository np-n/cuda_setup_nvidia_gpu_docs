### Environment setup for NVIDIA GPU

#### 1. Check driver version supported
```
ubuntu-drivers devices
```
Output:

```commandline
WARNING:root:_pkg_get_support nvidia-driver-515: package has invalid Support PBheader, cannot determine support level
WARNING:root:_pkg_get_support nvidia-driver-525-server: package has invalid Support PBheader, cannot determine support level
WARNING:root:_pkg_get_support nvidia-driver-515-server: package has invalid Support PBheader, cannot determine support level
WARNING:root:_pkg_get_support nvidia-driver-525: package has invalid Support PBheader, cannot determine support level
WARNING:root:_pkg_get_support nvidia-driver-530: package has invalid Support PBheader, cannot determine support level
WARNING:root:_pkg_get_support nvidia-driver-510: package has invalid Support PBheader, cannot determine support level
== /sys/devices/pci0000:00/0000:00:03.1/0000:0a:00.0 ==
modalias : pci:v000010DEd00002204sv00001043sd000087B3bc03sc00i00
vendor   : NVIDIA Corporation
driver   : nvidia-driver-515 - distro non-free
driver   : nvidia-driver-525-server - distro non-free
driver   : nvidia-driver-470 - distro non-free recommended
driver   : nvidia-driver-470-server` - distro non-free
driver   : nvidia-driver-515-server - distro non-free
driver   : nvidia-driver-525 - third-party non-free
driver   : nvidia-driver-530 - distro non-free
driver   : nvidia-driver-510 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin

== /sys/devices/pci0000:00/0000:00:01.2/0000:01:00.0/0000:02:06.0/0000:06:00.0 ==
modalias : pci:v00008086d00002723sv00008086sd00000084bc02sc80i00
vendor   : Intel Corporation
manual_install: True
driver   : backport-iwlwifi-dkms - distro free
```

#### 2. Check  nvidia driver and cuda compatibility, and install the driver that is compatible to `cuda` version that you have required.
- https://docs.nvidia.com/deploy/cuda-compatibility/index.html, https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- Driver installation example:
  - We want to use `cuda 10.x` and `cuda 11.x` for our GPU accelerated computing, after checking the driver-cuda compatibility, we found that driver version `>= 450.80.02*` is compatible to both `cuda 10.x` and `cuda 11.x`.
  - Now, install nvidia-driver `nvidia-driver-470-server` 
    - `sudo apt-get install nvidia-driver-470-server`
#### 3. Now, restart the system and check the installed gpu driver
```
nvidia-smi
```

- From following `nivida-smi` output, we knew that nvidia-driver `470.182.03` is installed on our system.
```commandline
Mon May 22 23:22:30 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.182.03   Driver Version: 470.182.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:0A:00.0 Off |                  N/A |
|  0%   38C    P8    20W / 350W |     23MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1244      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      1324      G   /usr/bin/gnome-shell               12MiB |
+-----------------------------------------------------------------------------+
```

- It will show the higher cuda version having compatibility and gpu memory status.
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.182.03   Driver Version: 470.182.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:0A:00.0 Off |                  N/A |
|  0%   31C    P8     6W / 350W |     23MiB / 24268MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1270      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      1382      G   /usr/bin/gnome-shell               12MiB |
+-----------------------------------------------------------------------------+
```


### Now start to setup CUDA 
- Let's go for cuda 10.2
- Visit nvidia cuda-toolkit archive https://developer.nvidia.com/cuda-toolkit-archive and download `.run` file for cuda version , here we area going to download cuda 10.2 version
```
wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
```

- After completion of download run downloaded `.run` file
```
sudo sh cuda_10.2.89_440.33.01_linux.run
```
	- Note: On the Agreement section deselect/unselect driver  checkbox since driver 470 is installed.
Output: 
```
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-10.2/
Samples:  Installed in /home/zakipoint/, but missing recommended libraries

Please make sure that
 -   PATH includes /usr/local/cuda-10.2/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-10.2/lib64, or, add /usr/local/cuda-10.2/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-10.2/bin

Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-10.2/doc/pdf for detailed information on setting up CUDA.
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 440.00 is required for CUDA 10.2 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log
```
Now, cuda recommed to do following configuration.
```
Please make sure that
 -   PATH includes /usr/local/cuda-10.2/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-10.2/lib64, or, add /usr/local/cuda-10.2/lib64 to /etc/ld.so.conf and run ldconfig as root
 ```
---
#### After installation, Create `.bashrc` file and add following:
```
export PATH=“/usr/local/cuda-10.2/bin:$PATH”  
export LD_LIBRARY_PATH=“/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH”
```



#### Now, Download cudnn compatible to cuda version from 
- [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive), this will require login in nvidia.
- Now download following three cuda compatible cudnn for your os.
	- **[cuDNN Runtime Library for Ubuntu18.04 (Deb)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/Ubuntu18_04-x64/libcudnn7_7.6.5.32-1%2Bcuda10.2_amd64.deb)**  
	- **[cuDNN Developer Library for Ubuntu18.04 (Deb)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/Ubuntu18_04-x64/libcudnn7-dev_7.6.5.32-1%2Bcuda10.2_amd64.deb)**  
	- **[cuDNN Code Samples and User Guide for Ubuntu18.04 (Deb)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/Ubuntu18_04-x64/libcudnn7-doc_7.6.5.32-1%2Bcuda10.2_amd64.deb)**
-  Now, extract and install above three packages using `sudo dpkg -i <packaage_name>`
- Go to `cd /usr/src/cudnn_samples_v7/mnistCUDNN` path and
- clean and compile using  `sudo make clean && sudo make`, output:
```
rm -rf *o
rm -rf mnistCUDNN
Linking agains cublasLt = true
CUDA VERSION: 10020
TARGET ARCH: x86_64
HOST_ARCH: x86_64
TARGET OS: linux
SMS: 30 35 50 53 60 61 62 70 72 75
/usr/local/cuda/bin/nvcc -ccbin g++ -I/usr/local/cuda/include -I/usr/local/cuda/include -IFreeImage/include  -m64    -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o fp16_dev.o -c fp16_dev.cu
g++ -I/usr/local/cuda/include -I/usr/local/cuda/include -IFreeImage/include   -o fp16_emu.o -c fp16_emu.cpp
g++ -I/usr/local/cuda/include -I/usr/local/cuda/include -IFreeImage/include   -o mnistCUDNN.o -c mnistCUDNN.cpp
/usr/local/cuda/bin/nvcc -ccbin g++   -m64      -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o mnistCUDNN fp16_dev.o fp16_emu.o mnistCUDNN.o -I/usr/local/cuda/include -I/usr/local/cuda/include -IFreeImage/include  -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64 -lcublasLt -LFreeImage/lib/linux/x86_64 -LFreeImage/lib/linux -lcudart -lcublas -lcudnn -lfreeimage -lstdc++ -lm
```

- And finally run `./mnistCUDNN` and it will output Test passed! and classification result as follows.
```
cudnnGetVersion() : 7605 , CUDNN_VERSION from cudnn.h : 7605 (7.6.5)
Host compiler version : GCC 7.5.0
There are 1 CUDA capable devices on your machine :
device 0 : sms 82  Capabilities 8.6, SmClock 1740.0 Mhz, MemSize (Mb) 24268, MemClock 9751.0 Mhz, Ecc=0, boardGroupID=0
Using device 0

Testing single precision
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 0
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.002048 time requiring 100 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.009216 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.024576 time requiring 57600 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.028544 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.032768 time requiring 203008 memory
Resulting weights from Softmax:
0.0000000 0.9999399 0.0000000 0.0000000 0.0000561 0.0000000 0.0000012 0.0000017 0.0000010 0.0000000 
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 0.9999288 0.0000000 0.0000711 0.0000000 0.0000000 0.0000000 0.0000000 
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 0.9999820 0.0000154 0.0000000 0.0000012 0.0000006 

Result of classification: 1 3 5

Test passed!

Testing half precision (math in single precision)
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 0
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.002048 time requiring 100 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.008192 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.020512 time requiring 28800 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.027648 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.031744 time requiring 203008 memory
Resulting weights from Softmax:
0.0000001 1.0000000 0.0000001 0.0000000 0.0000563 0.0000001 0.0000012 0.0000017 0.0000010 0.0000001 
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 1.0000000 0.0000000 0.0000714 0.0000000 0.0000000 0.0000000 0.0000000 
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 1.0000000 0.0000154 0.0000000 0.0000012 0.0000006 

Result of classification: 1 3 5

Test passed!
```



----
#### To check version of CUDA
```
nvcc --version
```
Output:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Wed_Oct_23_19:24:38_PDT_2019
Cuda compilation tools, release 10.2, V10.2.89
```


-----

#### Remove cuda from system
To remove CUDA from your Ubuntu system, you can follow these steps:

1.  Open a terminal on your Ubuntu system.
    
2.  Check if the CUDA toolkit is installed by running the following command:
    
    `nvcc --version` 
    
    If CUDA is installed, you will see output similar to:
    
    pythonCopy code
    
    `nvcc: NVIDIA (R) Cuda compiler driver
    ...` 
    
3.  If CUDA is installed, proceed with the removal process. First, uninstall the CUDA toolkit by running the following command:
    
    arduinoCopy code
    
    `sudo apt-get --purge remove cuda` 
    
4.  Next, remove any leftover files and directories related to CUDA:
    
    bashCopy code
    
    `sudo rm -rf /usr/local/cuda*` 
    
5.  If you had added CUDA-related paths to your environment variables, you may also want to remove them. Open the `.bashrc` file in a text editor:
    
    bashCopy code
    
    `nano ~/.bashrc` 
    
6.  Look for any lines that export CUDA-related paths, such as `export PATH=/usr/local/cuda-X.X/bin:$PATH` or `export LD_LIBRARY_PATH=/usr/local/cuda-X.X/lib64:$LD_LIBRARY_PATH`. Remove these lines or comment them out by adding a `#` at the beginning of each line.
    
7.  Save the `.bashrc` file and exit the text editor.
    
8.  Finally, reload the modified `.bashrc` file to apply the changes:
    
    bashCopy code
    
    `source ~/.bashrc` 
    
9.  CUDA should now be removed from your Ubuntu system. You can verify this by running the `nvcc --version` command again. If CUDA is successfully removed, the command should not be found or display an error message.




---


#### Uninstall nvidia driver completely
**For Ubuntu 12.04-22.04**


- Search what packages from nvidia you have installed.
```
dpkg -l | grep -i nvidia
```
**except**  the package  `nvidia-common`  all other packages should be purged.

----------

- If you want to be sure that you will purge everything related to nvidia you can give this command
```
sudo apt-get remove --purge '^nvidia-.*'
```


But,
Above command will also remove the  `nvidia-common`  package and the  `nvidia-common`  package has as a dependency the  `ubuntu-desktop`  package.

So after above command you should also give the installation command for  `ubuntu-desktop`  package

```
sudo apt-get install ubuntu-desktop
```
