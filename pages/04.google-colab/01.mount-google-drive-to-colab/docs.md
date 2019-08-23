---
title: 'Mount Google Drive to Colab'
taxonomy:
    category:
        - docs
visible: true
---

```py
#@markdown This cell is to mount your Google Drive in Colaboratory. Run it and follow the instruction.

mount_point = '/content/gdrive' #@param {type:'string'}

import os
if os.path.isdir(mount_point):
  print(mount_point + ' has been already mounted.')
else:
  from google.colab import drive
  drive.mount(mount_point)
```