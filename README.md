

Prerequisites:
1. Anaconda/Miniconda with Python 3.8
2. Jupyter Notebook

# Note(s)
1. Update the Python environment when the dependencies is updated. This is to ensure reproducibility in different machines.

```bash
pip list --format=freeze > requirements.txt
```

2. Ensure you are in the right environment
* Anaconda/Miniconda:
```bash
conda env list
```

3. Isolate the Python environment of this project using [venv](https://python.land/virtual-environments/virtualenv). I have seperated and commit the configuration files.

```bash
python -m venv capstone-project
source capstone-project/bin/activate  # On MacOS/Linux
capstone-project\Scripts\activate.bat  # On Windows
```

4. Install GCloud CLI. To access to BigQuery and download the Ethereum dataset to perform extraction.
* https://cloud.google.com/sdk/docs/install (Install)
* https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev (Login)
* gcloud auth application-default login


# Instructions

wsl

https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl
https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

wget https://repo.continuum.io/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
conda activate capstone-project
nvidia-smi
nvcc --version
python --version

```bash
(capstone-project) tkhenghong@DESKTOP-SSLGQJT:/mnt/c/Users/Z690 GAMING X DDR4/Desktop/tkh-apu-master/capstone-project$ nvidia-smi
Fri May  3 01:18:27 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 552.22         CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4080 ...    On  |   00000000:01:00.0  On |                  N/A |
|  0%   42C    P8             18W /  320W |    1449MiB /  16376MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
(capstone-project) tkhenghong@DESKTOP-SSLGQJT:/mnt/c/Users/Z690 GAMING X DDR4/Desktop/tkh-apu-master/capstone-project$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
(capstone-project) tkhenghong@DESKTOP-SSLGQJT:/mnt/c/Users/Z690 GAMING X DDR4/Desktop/tkh-apu-master/capstone-project$ python --version
Python 3.9.19
(capstone-project) tkhenghong@DESKTOP-SSLGQJT:/mnt/c/Users/Z690 GAMING X DDR4/Desktop/tkh-apu-master/capstone-project$
```

1. Install the Python dependencies, with [mirrors**](https://charly-lersteau.com/blog/2019-11-24-faster-python-pip-install-mirrors/) :

```bash
conda create --name capstone-project python=3.9.19
conda activate capstone-project
pip install -r requirements.txt
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
conda install jupyter
conda install tensorflow-gpu
conda install keras
conda install numpy~=1.23.5

conda install chardet
conda install pandas

python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```


```bash
conda deactivate
conda remove -n capstone-project --all -y
```

2. Try to click Run All in you Python notebook.

3. That's it for now.