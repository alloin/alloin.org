---
title: 'All in one (source to edges)'
---

First, we create a conda enviroment for installing **caffe**
```
conda create -n caffe python=3.7
conda activate caffe
conda config --add channels anaconda
conda install caffe -c willyd
```

app.py
```py
import os
import tensorflow as tf
import sys
import datetime
import subprocess

import numpy as np

from os import listdir
from shutil import copyfile
from io import BytesIO
from PIL import Image

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

"""=================================================================================================================="""
fill_color = '#ffffff'  # your background

input_dir = 'D:/AI/tests/input/'
resized_dir = 'D:/AI/tests/resized/'
cut_dir = 'D:/AI/tests/cut/'
edges_dir = 'D:/AI/tests/edges/'
white_dir = 'D:/AI/tests/white/'
merge_dir = 'D:/AI/tests/merged/'
process_dir = 'D:/AI/tests/processed/'
output_dir = 'D:/AI/tests/output/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('directory created:'+output_dir)

if not os.path.exists(resized_dir):
    os.makedirs(resized_dir)
    print('directory created:'+resized_dir)

if not os.path.exists(edges_dir):
    os.makedirs(edges_dir)
    print('directory created:'+edges_dir)

if not os.path.exists(merge_dir):
    os.makedirs(merge_dir)
    print('directory created:'+merge_dir)

if not os.path.exists(white_dir):
    os.makedirs(white_dir)
    print('directory created:'+white_dir)

if not os.path.exists(cut_dir):
    os.makedirs(cut_dir)
    print('directory created:'+cut_dir)

if not os.path.exists(process_dir):
    os.makedirs(process_dir)
    print('directory created:'+process_dir)


print(str(len(os.listdir(input_dir))) + " input files found.")
print(str(len(os.listdir(cut_dir))) + " cut files found.")
print(str(len(os.listdir(resized_dir))) + " resized files found.")
print(str(len(os.listdir(edges_dir))) + " edges files found.")
print(str(len(os.listdir(white_dir))) + " white files found.")
print(str(len(os.listdir(merge_dir))) + " merged files found.")
print(str(len(os.listdir(process_dir))) + " processed files found.")
print(str(len(os.listdir(output_dir))) + " output files found.")

"""=================================================================================================================="""
#python tools/process.py --input_dir photos/dogs --operation resize --output_dir photos/dogs_resized
subprocess.call(["python", "process.py", "--input_dir", input_dir, "--output_dir", resized_dir, "--operation", "resize"])
"""=================================================================================================================="""
"""
!wget http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
!wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz

!mkdir mobile_net_model
!mkdir xception_model
!tar xvzf deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz -C mobile_net_model --strip=1
!tar xvzf deeplabv3_pascal_train_aug_2018_01_04.tar.gz -C xception_model --strip=1

!rm deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
!rm deeplabv3_pascal_train_aug_2018_01_04.tar.gz
!wget http://ohrly.net/dogs_dataset.rar
!apt install unrar
!unrar x /content/dogs_dataset.rar 'D:/AI/tests/1/'
"""
class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
        image: A PIL.Image object, raw input image.
        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """
        start = datetime.datetime.now()

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        end = datetime.datetime.now()

        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))

        return resized_image, seg_map

def drawSegment(baseImg, matImg, filename):
      width, height = baseImg.size
      dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
      for x in range(width):
                for y in range(height):
                    color = matImg[y,x]
                    (r,g,b) = baseImg.getpixel((x,y))
                    if color == 0:
                        dummyImg[y,x,3] = 0
                    else :
                        dummyImg[y,x] = [r,g,b,255]
      img = Image.fromarray(dummyImg)
      #img = img.convert("RGB")
      if img.mode in ('RGBA', 'LA'):
          background = Image.new(img.mode[:-1], img.size, fill_color)
          background.paste(img, img.split()[-1])
          img = background
      img.save(cut_dir+filename)
      print('saving to:'+cut_dir+filename)


inputFilePath = resized_dir
outputFilePath = cut_dir

if inputFilePath is None or outputFilePath is None:
    print("Bad parameters. Please specify input file path and output file path")
    exit()

#modelType = "xception_model"
modelType = "mobile_net_model"

MODEL = DeepLabModel(modelType)
print('model loaded successfully : ' + modelType)

"""=================================================================================================================="""

def run_visualization(fpath, filename):
    filepath = fpath + filename
    """Inferences DeepLab model and visualizes result."""
    try:
        print("Trying to open : " + filepath)
        # f = open(sys.argv[1])
        jpeg_str = open(filepath, "rb").read()
        orignal_im = Image.open(BytesIO(jpeg_str))
    except IOError:
        print('Cannot retrieve image. Please check file: ' + filepath)
        return

    print('running deeplab on image %s...' % filepath)
    resized_im, seg_map = MODEL.run(orignal_im)

    # vis_segmentation(resized_im, seg_map)
    drawSegment(resized_im, seg_map, filename)

"""=================================================================================================================="""

for filename in listdir(resized_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        filename_png = os.path.splitext(filename)[0] + '.png'
        input_path = os.path.join(resized_dir, filename)
        output_path = os.path.join(cut_dir, filename_png)
        if os.path.isfile(output_path): continue
        try:
            print('Processing: ' + filename)
            run_visualization(resized_dir, filename)
        except Exception as e:
            print('Error: ' + str(e))

"""=================================================================================================================="""

subprocess.call(["python", "process.py", "--input_dir", cut_dir, "--output_dir", edges_dir, "--operation", "edges"])

subprocess.call(["python", "process.py", "--input_dir", resized_dir, "--b_dir", edges_dir, "--output_dir", output_dir, "--operation", "combine"])

subprocess.call(["python", "split.py", "--dir", output_dir])
```