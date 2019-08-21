---
title: 'Convert Transparent pixels to white'
taxonomy:
    category:
        - docs
---

```py
from os import listdir
from shutil import copyfile
from PIL import Image

#target_dir = '/content/gdrive/My Drive/ai/out1/'

def removealpha(fpath, filename):
  image = Image.open(fpath+filename)
  image.convert("RGBA") # Convert this to RGBA if possible

  pixel_data = image.load()

  if image.mode == "RGBA":
    # If the image has an alpha channel, convert it to white
    # Otherwise we'll get weird pixels
    for y in xrange(image.size[1]): # For each row ...
      for x in xrange(image.size[0]): # Iterate through each column ...
        # Check if it's opaque
        if pixel_data[x, y][3] < 255:
          # Replace the pixel data with the colour white
          pixel_data[x, y] = (255, 255, 255, 255)

  # Resize the image thumbnail
  #image.thumbnail([resolution.width, resolution.height], Image.ANTIALIAS)
  #image.convert("RGB")
  print('removed transparent layer of '+filename)
  image.save(white_dir+filename) 
 
for filename in listdir(cut_dir):
    removealpha(cut_dir, filename)
```