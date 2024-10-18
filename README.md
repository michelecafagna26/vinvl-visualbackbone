# VinVL VisualBackbone

Original VinVL visual backbone with simplified APIs to easily extract features, boxes, object detections in a few lines of code.
This repo is based on [microsoft/scene_graph_benchmark](https://github.com/microsoft/scene_graph_benchmark) please refer that repo for further info about the benchmark

## Installation

Create your virtual environment an install the following dependencies according to your system specs.
- PyTorch 1.7
- torchvision

Then run:
```bash
!git clone https://github.com/michelecafagna26/vinvl-visualbackbone.git
%cd /content/vinvl-visualbackbone
!pip install -r requirements.txt --quiet
%cd /content
```

```bash
!pip install torch==1.13.1 torchvision==0.14.1 numpy==1.21.6 --quiet
```

```bash
%cd /content/vinvl-visualbackbone/scene_graph_benchmark
!python setup.py build develop --quiet
%cd /content
```

```bash
%cd /content/vinvl-visualbackbone/scene_graph_benchmark
!mkdir -p models
%cd models
# download from the huggingface model hub
!git lfs install # if not installed
!git clone https://huggingface.co/michelecafagna26/vinvl_vg_x152c4
%cd /content
```



You can check the original [INSTALL.md](INSTALL.md) for alternative installation options

## Model download

Download the model **before** running your code.

```bash
%cd /content/vinvl-visualbackbone/scene_graph_benchmark
!mkdir -p models
%cd models
# download from the huggingface model hub
!git lfs install # if not installed
!git clone https://huggingface.co/michelecafagna26/vinvl_vg_x152c4
%cd /content
```
<!-- ## Alternative Model download (links might be broken )**(deprecated)**

If not present, the model is automatically downloaded. However, *it can take a while, so it's advised to manually download it **before** running your code*
```bash

mkdir -p scene_graph_benchmark/models/vinvl_vg_x152c4
cd scene_graph_benchmark/models/vinvl_vg_x152c4

# download the model
wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth

# downlaod the labelmap
wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json
```
-->

## Quick start: feature extraction

```python

img_file = "/content/vinvl-visualbackbone/scene_graph_benchmark/demo/woman_fish.jpg"
feature_extractor = VinVLVisualBackbone()
dets = feature_extractor(img_file)
dets.keys()
```
`dets` contains the following keys: ```["boxes", "classes", "scores", "features", "spatial_features"]```
You can obtain the full VinVL's visual features by concatenating "features" and "spatial_features"

```python
import numpy as np

v_feats = np.concatenate((dets['features'],  dets['spatial_features']), axis=1)
v_feats.shape   # (num_boxes, 2054)
```
----
## Demo
Coming Soon!

----
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

@inproceedings{zhang2021vinvl,
  title={Vinvl: Revisiting visual representations in vision-language models},
  author={Zhang, Pengchuan and Li, Xiujun and Hu, Xiaowei and Yang, Jianwei and Zhang, Lei and Wang, Lijuan and Choi, Yejin and Gao, Jianfeng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5579--5588},
  year={2021}
}
```
