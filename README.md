# PythonMMReader
Python code for reading micro-manager tiff stacks

```
from reader import MicroManagerStack

path = '/Users/henrypinkard/Desktop/mmtest_1'

dataset = MicroManagerStack(path)
pixels, metadata = dataset.read_image(channel_index=0, z_index=0, t_index=0, pos_index=0, read_metadata=True)
```
