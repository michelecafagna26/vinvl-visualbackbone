# VinVL VisualBackbone

Original VinVL visual backbone with simplified APIs to easily extract features, boxes, object detections in a few lines of code.
This repo is based on [microsoft/scene_graph_benchmark](https://github.com/microsoft/scene_graph_benchmark) please refer that repo for further info about the benchmark

## Installation

Create your virtual environment an install the following dependencies according to your system specs.
- PyTorch 1.7
- torchvision

Then run:
```bash
# install the requirements
pip install -r requirements.txt

# glone this repo
git clone git@github.com:michelecafagna26/vinvl_visualbackbone.git

# install Scene Graph Detection with the VisualBackbone apis
pyton setup.py build develop
```
You can check the original [INSTALL.md](INSTALL.md) for alternative installation options

----
## Quick start

```python
from scene_graph_benchmark.wrappers import VinVLVisualBackbone

img_file = "scene_graph_bechmark/demo/woman_fish.jpg"

detector = VinVLVisualBackbone()

dets = detector(img_file)

```
`dets` contains the following keys: ["boxes", "classes", "scores", "features", "spatial_features"]
You can obtain the full VinVL's visual features by concatenating the "features" and the "spatial_features"

```python
import numpy as np

v_feats = np.concatenate((dets['features'],  dets['spatial_features']), axis=1)
# v_feats.shape = (num_boxes, 2054)
```
