---
title: 'Images to Movie'
taxonomy:
    category:
        - docs
visible: true
---

```
import cv2
import numpy as np
import glob

sourcemap = 'C:/images/*.jpg'

fps = 120

img_array = []
for filename in glob.glob(sourcemap):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
```