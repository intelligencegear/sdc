# **Install Environment Tricks**
First of all, assuming we are using Ubuntu 16.04.

### **Docker install acceleration using aliyun**

* After registering a aliyun account, we can use aliyun docker acceleration service.

  ```shell
  sudo tee /etc/docker/daemon.json <<-'EOF'
  {
    "registry-mirrors": ["<your accelerate address>"]
  }
  EOF
  sudo systemctl daemon-reload
  sudo systemctl restart docker
  ```

* Now we can follow Udacity install guide
  ```shell
  docker pull udacity/carnd-term1-starter-kit
  docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
  ```

### **Miniconda install acceleration**
* Assuming we have install Nvidia drivers, cuda&cudnn releted libraries correctly.

* Download latest Miniconda from https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/. Install it.

* Change install software dependencies(GPU version) like below:
```shell
name: carnd-term1
channels:
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
dependencies:
    - python==3.5.2
    - numpy
    - matplotlib
    - jupyter
    - opencv3
    - pillow
    - scikit-learn
    - scikit-image
    - scipy
    - h5py
    - eventlet
    - flask-socketio
    - seaborn
    - pandas
    - ffmpeg
    - imageio=2.1.2
    - pyqt=4.11.4
    - pip:
        - moviepy
        - https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
        - keras==1.2.1
```

* Add snippet below into ~/.pip/pip.conf

  ```shell
  [global]

  index-url = https://pypi.tuna.tsinghua.edu.cn/simple
  ```
* Now we can follow Udacity guide to create **Miniconda** environment to start our project.

  ```shell
  conda env create -f environment-gpu.yml
  ```


### Reference
https://yq.aliyun.com/articles/29941

https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

http://www.jianshu.com/p/502638407add
