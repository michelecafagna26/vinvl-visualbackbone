# VinVL VisualBackbone

Original VinVL visual backbone with simplified APIs to easily extract features, boxes, object detections in a few lines of code.
This repo is based on [microsoft/scene_graph_benchmark](https://github.com/microsoft/scene_graph_benchmark) please refer that repo for further info about the benchmark

## Installation

Create your virtual environment an install the following dependencies according to your system specs.
- PyTorch 1.7
- torchvision

Then run:
```bash
# good practice
pip install --upgrade pip

# install the requirements
pip install -r requirements.txt

# glone this repo
git clone git@github.com:michelecafagna26/vinvl-visualbackbone.git

# install Scene Graph Detection with the VisualBackbone apis
pyton setup.py build develop
```
You can check the original [INSTALL.md](INSTALL.md) for alternative installation options

## Model downloading (strongly suggested)

If not present, the model is automatically downloaded. However, *it can take a while, so it's advised to manually download it*
```bash

mkdir -p scene_graph_benchmark/models/vinvl
cd scene_graph_benchmark/models/vinvl

# download the model
wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth

# downlaod the labelmap
wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json
```
----
## Quick start: feature extraction

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
## Citations
Please consider citing the original project and the VinVL paper
```BibTeX
@misc{han2021image,
      title={Image Scene Graph Generation (SGG) Benchmark}, 
      author={Xiaotian Han and Jianwei Yang and Houdong Hu and Lei Zhang and Jianfeng Gao and Pengchuan Zhang},
      year={2021},
      eprint={2107.12604},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

inproceedings{zhang2021vinvl,
  title={Vinvl: Revisiting visual representations in vision-language models},
  author={Zhang, Pengchuan and Li, Xiujun and Hu, Xiaowei and Yang, Jianwei and Zhang, Lei and Wang, Lijuan and Choi, Yejin and Gao, Jianfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5579--5588},
  year={2021}
}
```
