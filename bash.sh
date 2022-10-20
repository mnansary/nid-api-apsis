#!/bin/sh
pip install paddlepaddle==2.4.0rc0
pip install opencv-python
pip install shapely
pip install pyclipper
pip install scikit-image
pip install imgaug
pip install lmdb
pip install tqdm
pip install attrdict
pip install git+https://github.com/mnansary/PaddleOCR.git --verbose
pip install torch torchvision  --extra-index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime
pip install termcolor
pip install gdown
python weights/download.py
python setup_check.py
echo succeeded
