Client side:

sudo apt-get update
sudo apt-get upgrade


python3 -m pip install torch --no-dependencies --break-system-packages
python3 -m pip install torchvision --no-dependencies --break-system-packages

sudo apt-get install libopenblas-dev m4 libblas-dev cmake libatlas-base-dev gfortran libffi-dev libavformat-dev libavdevice-dev libjpeg-dev

python3 -m pip install -U pip setuptools


PC/Server Side
Install PyTorch
Command installation is sufficient, installing the CUDA version is also an option

python -m pip install torch torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

python -m pip install -U pip setuptools
python -m pip install lz4~=3.0.2 msgpack~=1.0.0 phe~=1.4.0 scipy~=1.4.1 syft-proto~=0.2.5.a1 tblib~=1.6.0 websocket-client~=0.57.0 websockets~=8.1.0 zstd~=1.4.4.0 Flask~=1.1.1 tornado==4.5.3 flask-socketio~=4.2.1 lz4~=3.0.2 Pillow~=6.2.2 requests~=2.22.0 numpy~=1.18.1


to run:

python main.py --server_ip <ip> --server_port <port>
