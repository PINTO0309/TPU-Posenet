# TPU-Posenet

August 18, 2019 Under construction.

## １．Introduction procedure
**Common procedures for devices**
```bash
$ sudo apt-get install -y python3-pip
$ sudo pip3 install pip --upgrade
$ sudo pip3 install imutils

sudo apt-get update

$ wget https://dl.google.com/coral/edgetpu_api/edgetpu_api_latest.tar.gz -O edgetpu_api.tar.gz --trust-server-names
$ tar xzf edgetpu_api.tar.gz
$ sudo edgetpu_api/install.sh

$ git clone https://github.com/PINTO0309/TPU-Posenet.git
$ cd TPU-Posenet.git
$ models/download.sh
$ media/download.sh
```

**Only RaspberryPi**
```bash
### Only Raspbian Buster ############################################################
$ cd /usr/local/lib/python3.7/dist-packages/edgetpu/swig/
$ sudo cp _edgetpu_cpp_wrapper.cpython-35m-arm-linux-gnueabihf.so _edgetpu_cpp_wrapper.cpython-37m-arm-linux-gnueabihf.so
### Only Raspbian Buster ############################################################

$ cd ~/TPU-Posenet
$ sudo raspi-config
```
![01](media/01.png)  
![02](media/02.png)  
![03](media/03.png)  
![04](media/04.png)  
![05](media/05.png)  
![06](media/06.png)  
