---
title: 'Image-to-Image Translation'
media_order: tBTUqNM.png
taxonomy:
    category:
        - docs
process:
    markdown: true
    twig: true
---

#Image-to-Image Translation with Conditional Adversarial Nets

In this part we will process a Dataset, train and test our Neural Network and make a javascript widget for it. 

We will use [affinelayer's pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) port of [Philipe Isola's pix2pix](https://github.com/phillipi/pix2pix)
As dataset, we will use [Jonathan Krause's Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

##Prerequisites
```
pip install tensorflow-gpu
pip install keyboard
pip install Pillow
pip install opencv-python
```
clone the tools and make a new folder called *input*
```
git clone https://github.com/alloin/tools
cd tools
mkdir input
```
extract the Dataset into the *input* folder, be sure there's **no subfolders**
run **app.py**
```
python app.py
```
The script will make folders, resize the images, remove the backgrounds, converts  images to edges and then combines the resized images with the edges.
Finally, the images are put into 2 folders, for training and testing.

After a while (depending on the amount of pictures), you should have 2 folders (*train* and *val*) inside the **output** folder, both containing combined sets of the images

Now that our Dataset is processed, we can finally start training our Neural Network.

We start by cloning the pix2pix-tensorflow repo
```
git clone https://github.com/affinelayer/pix2pix-tensorflow.git
cd pix2pix-tensorflow
```
To train our first model, all you have to do is start pix2pix.py with the correct arguments
```
python pix2pix.py --mode train --output_dir cars_train --max_epochs 1000 --input_dir cars/train --which_direction BtoA 
```
While training, you can look at the current progress of the training through *Tensorboard*
If you have tensorboard installed, open a new terminal in the current directory and type 

```
tensorboard --logdir=cars_train
```
Now you can go to http://localhost:6006 to check your training, you should see something simular to this:
![](tBTUqNM.png?resize=400,200)![](tvNhtnC.png?resize=400,200)

```
python pix2pix.py --mode train --output_dir cars_train --max_epochs 1000 --input_dir cars/train --checkpoint cars_train --which_direction BtoA 
```
```
python pix2pix.py --mode test --output_dir cars_test --input_dir cars/val --checkpoint cars_train
```
```
python pix2pix.py --mode export --output_dir cars_export --input_dir cars_train --checkpoint cars_train
```

to be continued...


@article{pix2pix2016,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  journal={arxiv},
  year={2016}
}