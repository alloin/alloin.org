---
title: 'Google Colab Parameters'
---

```py
#@markdown This cell contains some parameters. Update and run it.

input_dir = '/My Drive/ai/dogs_resized/' #@param {type:"string"}
input_dir = mount_point + input_dir

cut_dir = '/My Drive/ai/dogs_bgremoved/' #@param {type:"string"}
cut_dir = mount_point + cut_dir

white_dir = '/My Drive/ai/dogs_whitebg/' #@param {type:"string"}
white_dir = mount_point + white_dir

merge_dir = '/My Drive/ai/dogs_channel/' #@param {type:"string"}
merge_dir = mount_point + merge_dir

process_dir = '/My Drive/ai/dogs_processed/' #@param {type:"string"}
process_dir = mount_point + process_dir

output_dir = '/My Drive/ai/dogs_edges/' #@param {type:"string"}
output_dir = mount_point + output_dir

import os

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print('directory created:'+output_dir)

if not os.path.exists(merge_dir):
    os.makedirs(merge_dir)
    print('directory created:'+merge_dir)

if not os.path.exists(white_dir):
    os.makedirs(white_dir)
    print('directory created:'+white_dir)

if not os.path.exists(cut_dir):
    os.makedirs(cut_dir)
    print('directory created:'+cut_dir)


print(str(len(os.listdir(input_dir))) + " input files found.")
```