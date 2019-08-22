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
![9kFks0I.png](9kFks0I.png?resize=600,300)

Now that our Dataset is processed, we can finally start training our Neural Network.

We start by cloning the pix2pix-tensorflow repo
```
git clone https://github.com/affinelayer/pix2pix-tensorflow.git
cd pix2pix-tensorflow
```
To train our first model, copy/paste the **output** folder to the **pix2pix-tensorflow** folder, rename it to **cars** and start pix2pix.py with the correct arguments
```
python pix2pix.py --mode train --output_dir cars_train --max_epochs 1000 --input_dir cars/train --which_direction BtoA 
```
If everything is installed correctly, you should get a simular output:
```PowerShell
progress  epoch 16  step 3674  image/sec 10.7  remaining 1870m
discrim_loss 0.04561677
gen_loss_GAN 7.569011
gen_loss_L1 0.1982678
```
As you can see, training this Dataset for 1000 epochs on a ***GTX 1080 ti*** takes about 31 hours to complete.

gen_loss_L1 is the difference between your actual training and your goal. Actually it is the mean reduce of it. You eventually want this close to 0.

```gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))```

discrim_loss and gen_loss_GAN are fighting against each other. This is a mathematical artifact.
discrim_loss is the measure of the training that aims to identify outputs as fakes.
That measure is used to train what we call the discriminator.

```discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))```

While gen_loss_GAN is the measure of the training that aims to identify outputs as real.

```gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))```

Both gen_loss_GAN and gen_loss_L1 are combined to train what we call the generator.
Deep learning is somehow finding a needle in a haystack.
gen_loss_L1 gives a simple way that converges fast.
But GAN can help to go further in image details that gen_loss_L1 will hardly discover.

The combination used by default in affinelayer/pix2pix.py is something like ```100*gen_loss_L1 + gan```
That means in a first phase, gen_loss_L1 will decrease, and probably gan will increase.
But then, when gen_loss_L1~gan you will probably see gan start decreasing (and so discrim_loss will increase).
That means the generator starts winning over the discriminator.

While training, you can look at graphs and examples of the current progress of the training through *Tensorboard* 
If you have tensorboard installed, open a new terminal in the current directory and type 

```
tensorboard --logdir=cars_train
```
Now you can go to http://localhost:6006 to check your training, you should see something simular to this:
![tBTUqNM.png](tBTUqNM.png?resize=600,300)![tvNhtnC.png](tvNhtnC.png?resize=600,300)

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