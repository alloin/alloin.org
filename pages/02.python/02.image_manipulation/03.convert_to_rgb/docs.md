---
title: 'Convert to RGB'
taxonomy:
    category:
        - docs
---

```py
from os import listdir
from PIL import Image


for filename in os.listdir(white_dir):
  png = Image.open(white_dir+filename)
  png.load() # required for png.split()

  background = Image.new("RGB", png.size, (255, 255, 255))
  background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

  background.save(process_dir+filename)
  print('processed and saved ' +process_dir+filename)
```