* content
{:toc}
# **BirdsEye Billiards Converter**

![image-20231113124848618](/imgs/image-20231113124848618.png)



## 프로젝트 구성원

team **마!**

[팀장 하민수](https://github.com/gkalstn000)

[팀원 김건기](https://github.com/KEON-KIM)

[팀원 백승헌](https://github.com/SeungHune)

[팀원 조용일](https://github.com/Choyongil)

지도교수 마상백



## 개발 환경

Machine

* CPU : intel i9-9900k

* GPU : RTX 2060 super

* MEM : 32GB

* Hard : 250GB

OS : ubuntu 18.04

Python : 3.7.6

CUDA : 10.0

tensorflow : 1.15

## 개발 환경 구축

tensorflow 1.15와 호환되는 CUDA는 10.0 미만 버전이기 때문에 최신 CUDA가 아닌 10.0 버전을 설치.

### STEP 1. 기존에 설치된 CUDA 제거

> ```bash
> sudo apt-get purge nvidia* && sudo apt-get autoremove && sudo apt-get autoclean && sudo rm -rf /usr/local/cuda*
> ```

> ```bash
> /usr/local
> ```
>
> 폴더에서  **CUDA** 와 관련된 모든 폴더 삭제

### STEP 2. Reboot

> ```bash
> sudo reboot
> ```

### STEP 3. Add NVIDIA Package repositories

> ```bash
> # Add NVIDIA package repositories
> wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
> sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
> sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
> sudo apt-get update
> wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
> sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
> sudo apt-get update
> ```

### STEP 4. Install NVIDIA driver

> ```bash
> sudo apt-get install --no-install-recommends nvidia-driver-418
> ```

### STEP 5. Reboot

>```bash
>sudo reboot
>```

### STEP 6. Install Runtime & Development Libraries (cuDNN)

>```bash
>sudo apt-get install --no-install-recommends cuda-10-0 libcudnn7=7.6.2.24-1+cuda10.0 libcudnn7-dev=7.6.2.24-1+cuda10.0
>```

### STEP 7. Install TensorRT

> ```bash
> sudo apt-get install -y --no-install-recommends libnvinfer5=5.1.5-1+cuda10.0 libnvinfer-dev=5.1.5-1+cuda10.0
> sudo reboot
> ```

### STEP 8. 설치 된 NVIDIA driver 확인

> ```bash
> nvidia-smi
> nvcc --version
> ```

### STEP 9. CUDA 환경변수 설정

> ```bash
> sudo vi ~/.bashrc
> ##############아래항목 맨 아래 추가##############
> export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
> export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
> #############################################
> #편집모드 종료 후
> source ~/.bashrc
> ```

### STEP 10. Anaconda 설치

[아나콘다 홈페이지](https://www.anaconda.com/distribution/)로 가서 anaconda download

> ```bash
> cd Download #navigate dir which you downloaded anaconda 
> bash Anaconda3-2019.10-Linux-x86_64.sh
> ```
>

### STEP 11. 가상환경 설치

> ```bash
> conda create -n [env_name] # create virtual environment
> source activate [env_name] # activate virtual environment
> source deactivate # deactivate virtual environment
> ```

**conda 가상 환경**이란 내가 작업할 프로젝트만을 위해 가상으로 공간을 할당하는 것입니다.  

예를들면 어떤 프로젝트는 텐서플로우 2.x 버전을 사용해야하고 어떤 프로젝트는 1.1x 버전을 사용해야 할 때, 한 머신에서 두 작업을 수행하려면 계속 텐서플로우를 지웠다 깔았다 반복을 해야하는 상황이 나옵니다.  

하지만 **conda 가상환경**은 내가 원하는 공간을 만들어 왔다갔다하며 두 작업을 모두 수행할 수 있게 해줍니다.

**[env_name]** 에 가상환경 이름을 할당해 주시면 됩니다.

example

> `conda create -n capstone`

### STEP 12. Tensorflow 1.15 설치

> ```bash
> pip install --upgrade tensorflow-gpu==1.15.0
> ```

### STEP 13. Tensorflow 설치 확인

anaconda spyder(python editor) 실행

> ```bash
> spyder
> ```

tensorflow 설치 확인

> ```python
> import tensorflow as tf
> 
> tf.test.is_gpu_available(
>     cuda_only=False,
>     min_cuda_compute_capability=None
> )
> 
> #result
> """
> 2020-04-15 23:22:22.199264: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
> 2020-04-15 23:22:22.219929: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
> 2020-04-15 23:22:22.220389: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x557100422970 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
> 2020-04-15 23:22:22.220400: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
> 2020-04-15 23:22:22.221619: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
> 2020-04-15 23:22:22.315925: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
> 2020-04-15 23:22:22.316348: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5571004cc1c0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
> 2020-04-15 23:22:22.316362: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce RTX 2060 SUPER, Compute Capability 7.5
> 2020-04-15 23:22:22.316501: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
> 2020-04-15 23:22:22.316810: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
> name: GeForce RTX 2060 SUPER major: 7 minor: 5 memoryClockRate(GHz): 1.665
> pciBusID: 0000:01:00.0
> 2020-04-15 23:22:22.317038: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
> 2020-04-15 23:22:22.317809: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
> 2020-04-15 23:22:22.318503: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
> 2020-04-15 23:22:22.318714: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
> 2020-04-15 23:22:22.319592: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
> 2020-04-15 23:22:22.320412: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
> 2020-04-15 23:22:22.322463: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
> 2020-04-15 23:22:22.322530: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
> 2020-04-15 23:22:22.322798: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
> 2020-04-15 23:22:22.323052: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
> 2020-04-15 23:22:22.323074: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
> 2020-04-15 23:22:22.323741: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
> 2020-04-15 23:22:22.323750: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
> 2020-04-15 23:22:22.323753: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
> 2020-04-15 23:22:22.323810: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
> 2020-04-15 23:22:22.324140: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:983] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
> 2020-04-15 23:22:22.324413: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/device:GPU:0 with 6823 MB memory) -> physical GPU (device: 0, name: GeForce RTX 2060 SUPER, pci bus id: 0000:01:00.0, compute capability: 7.5)
> Out[1]: True
> """
> ```

긴 줄이 나오고 True가 뜨면 CUDA & CUDNN 및 tensorflow 설치가 완료.

## Object Detection

![image-20231113124711908](/imgs/objecd.png)

Faster RCNN Inception model을 활용해 Table edge, ball 좌표값 확보.



## Transform image to graphic

![image-20231113125135815](/imgs/image-20231113125135815.png)

Raw 좌표값을 수직 view 로 transform 하기 위해 자체 알고리즘 개발.

* openCV [warpAffine](https://opencv-python.readthedocs.io/en/latest/doc/10.imageTransformation/imageTransformation.html) 변환 라이브러리보다 더 높은 정밀도

- <iframe src="https://www.youtube.com/embed/qnMxNfZViCY" title="" frameborder="0" style="margin: 0 auto; display: block;" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

- **좌**: openCV `warpAffine()` 라이브러리, **우**: 자체 개발 선형 변환 알고리즘

## Reference

[CUDA 10.0 설치]([https://teddylee777.github.io/linux/CUDA-%EC%9D%B4%EC%A0%84%EB%B2%84%EC%A0%84-%EC%82%AD%EC%A0%9C%ED%9B%84-%EC%9E%AC%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0](https://teddylee777.github.io/linux/CUDA-이전버전-삭제후-재설치하기))

[Object Detection 학습](https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d#862a)

[video detect](https://github.com/tensorflow/models/issues/6684)
