
#Reference
#Setup environment on Google cloud instance 

https://cloud.google.com/compute/docs/gpus/add-gpus

1. Change Quota
===============
https://console.cloud.google.com/iam-admin/quotas?project=cs281-197703

	a) Select us-west and edit quota 
		Google Compute Engine API
		NVIDIA K80 GPUs	us-west1	0 / 1
	b.) You will get an email from google upon approval of this request

2.) Configure the GPU instance 
==============================
	https://medium.com/google-cloud/using-a-gpu-tensorflow-on-google-cloud-platform-1a2458f42b0
	
	Startup script
	==============
	{
	  "canIpForward": false,
	  "cpuPlatform": "Unknown CPU Platform",
	  "creationTimestamp": "2018-03-15T13:48:46.883-07:00",
	  "deletionProtection": false,
	  "description": "",
	  "disks": [
		{
		  "autoDelete": true,
		  "boot": true,
		  "deviceName": "gpu-instance-tf",
		  "guestOsFeatures": [
			{
			  "type": "VIRTIO_SCSI_MULTIQUEUE"
			}
		  ],
		  "index": 0,
		  "interface": "SCSI",
		  "kind": "compute#attachedDisk",
		  "licenses": [
			"projects/ubuntu-os-cloud/global/licenses/ubuntu-1604-xenial"
		  ],
		  "mode": "READ_WRITE",
		  "source": "projects/cs281-197703/zones/us-west1-b/disks/gpu-instance-tf",
		  "type": "PERSISTENT"
		}
	  ],
	  "guestAccelerators": [
		{
		  "acceleratorCount": 1,
		  "acceleratorType": "https://www.googleapis.com/compute/beta/projects/cs281-197703/zones/us-west1-b/acceleratorTypes/nvidia-tesla-k80"
		}
	  ],
	  "id": "3839665378603524801",
	  "kind": "compute#instance",
	  "labelFingerprint": "42WmSpB8rSM=",
	  "machineType": "projects/cs281-197703/zones/us-west1-b/machineTypes/n1-standard-2",
	  "metadata": {
		"fingerprint": "70PC5ZJHe6M=",
		"items": [
		  {
			"key": "startup-script",
			"value": "#!/bin/bash\necho \"Checking for CUDA and installing.\"\n# Check for CUDA and try to install.\nif ! dpkg-query -W cuda-8-0; then\n  # The 16.04 installer works with 16.10.\n  curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb\n  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb\n  apt-get update\n  apt-get install cuda-8-0 -y\nfi\n# Enable persistence mode\nnvidia-smi -pm 1"
		  }
		],
		"kind": "compute#metadata"
	  },
	  "name": "gpu-instance-tf",
	  "networkInterfaces": [
		{
		  "accessConfigs": [
			{
			  "kind": "compute#accessConfig",
			  "name": "External NAT",
			  "networkTier": "PREMIUM",
			  "type": "ONE_TO_ONE_NAT"
			}
		  ],
		  "fingerprint": "ofVZMZ-97e4=",
		  "kind": "compute#networkInterface",
		  "name": "nic0",
		  "network": "projects/cs281-197703/global/networks/default",
		  "networkIP": "10.138.0.2",
		  "subnetwork": "projects/cs281-197703/regions/us-west1/subnetworks/default"
		}
	  ],
	  "scheduling": {
		"automaticRestart": true,
		"onHostMaintenance": "TERMINATE",
		"preemptible": false
	  },
	  "selfLink": "projects/cs281-197703/zones/us-west1-b/instances/gpu-instance-tf",
	  "serviceAccounts": [
		{
		  "email": "649900328375-compute@developer.gserviceaccount.com",
		  "scopes": [
			"https://www.googleapis.com/auth/devstorage.read_only",
			"https://www.googleapis.com/auth/logging.write",
			"https://www.googleapis.com/auth/monitoring.write",
			"https://www.googleapis.com/auth/servicecontrol",
			"https://www.googleapis.com/auth/service.management.readonly",
			"https://www.googleapis.com/auth/trace.append"
		  ]
		}
	  ],
	  "startRestricted": false,
	  "status": "TERMINATED",
	  "tags": {
		"fingerprint": "6smc4R4d39I=",
		"items": [
		  "http-server",
		  "https-server"
		]
	  },
	  "zone": "projects/cs281-197703/zones/us-west1-b"
	}

3.) Install packages 
====================
	(http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

	nvidia-smi
	echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
	echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
	echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
	source ~/.bashrc
	sudo apt-get install cuda 9.0
	tar -xvf cudnn-9.0-linux-x64-v7.tar
	sudo cp cuda/lib64/* /usr/local/cuda/lib64/
	sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
	sudo apt-get install python3-dev python3-pip libcupti-dev unzip python3-tk
	which pip3
	sudo pip3 install tensorflow-gpu
	

NVIDIA-SMI 
==========

gpur@gpu-instance-tf:~/workspace$ nvidia-smi
Sun Mar 18 04:57:52 2018
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.30                 Driver Version: 390.30                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla K80           On   | 00000000:00:04.0 Off |                    0 |
| N/A   70C    P8    35W / 149W |     15MiB / 11441MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1564      G   /usr/lib/xorg/Xorg                            15MiB |
+-----------------------------------------------------------------------------+

	
	Test: 
	=====
	import tensorflow as tf
	with tf.device('/cpu:0'):
		a_c = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a-cpu')
		b_c = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b-cpu')
		c_c = tf.matmul(a_c, b_c, name='c-cpu')
	with tf.device('/gpu:0'):
		a_g = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a-gpu')
		b_g = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b-gpu')
		c_g = tf.matmul(a_g, b_g, name='c-gpu')
	with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
		print (sess.run(c_c))
		print (sess.run(c_g))
	print ('DONE!')
	
	
	
	gpur@gpu-instance-tf:~/workspace/image_captioning$ python3 main.py --phase=train --load_cnn --cnn_model_file=vgg16_no_fc.npy [--train_cnn]
2018-03-18 04:49:56.041191: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-03-18 04:49:56.127324: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-03-18 04:49:56.127649: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties:
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:00:04.0
totalMemory: 11.17GiB freeMemory: 11.09GiB
2018-03-18 04:49:56.127676: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2018-03-18 04:49:56.395126: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10750 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)


4.) Additional packages 
=======================

	# https://pypi.python.org/pypi/opencv-contrib-python
	sudo pip3  install opencv-contrib-python
	# http://www.nltk.org/install.html
	sudo pip3  install -U nltk
	# https://scipy.org/install.html
	sudo pip3 install -U numpy scipy matplotlib ipython jupyter pandas sympy nose
	# https://pypi.python.org/pypi/tqdm
	sudo pip3 install -U tqdm
	sudo pip3 install scikit-image

	---- On Anaconda 3 -x64
	C:\Anaconda3\Scripts\activate.bat C:\Anaconda3
	conda install -c conda-forge tensorflow
	conda install -c conda-forge tensorboard 
	conda install -c aaronzs tensorflow-tensorboard
	conda install -c conda-forge opencv 
	conda install -c anaconda scikit-image tqdm nltk pandas
	conda install -c anaconda numpy scipy matplotlib ipython jupyter sympy nose
	conda install pillow
	conda update dask
	
	
5.) Setup data folders
======================
	Download: http://cocodataset.org/#download
	
	Resource punkt not found.
	Please use the NLTK Downloader to obtain the resource:
	>>> import nltk
	>>> nltk.download('punkt')

	

6.) Execute
===========
	Run the training phase
	python main.py --phase=train --load_cnn --cnn_model_file='./vgg16_no_fc.npy' [--train_cnn]
	
	Visualize the summary dump from tensorboard 
	tensorboard --logdir=summary\
	
------------------------- ISSUE -----------------------------
- Issue: Unable to load the tokenizer in pycoco folders
	

------------------------ Metrics -----------------------------
- Time on Cloud VM: 2018-03-18 22:07:14.627258: 
Epoch 0/100
022/11290 in 01 minutes @ 3.00 s/iters
046/11290 in 02 minutes @ 2.79 s/it [02:08<8:43:33]
125/11290 in 05 minutes @ 2.44 s/it [05:04<7:33:14]
264/11290 in 10 minutes @ 2.28 s/it [10:01<6:59:01]
998/11290 in 35 minutes @ 2.07 s/it [34:29<5:55:43]


	
	