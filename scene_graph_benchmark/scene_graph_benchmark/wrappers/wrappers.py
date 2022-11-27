from scene_graph_benchmark.AttrRCNN import AttrRCNN
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.miscellaneous import set_seed
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer

from scene_graph_benchmark.wrappers.utils import cv2Img_to_Image, encode_spatial_features

import torch
import json
import cv2
from pathlib import Path
from clint.textui import progress
import requests


BASE_PATH = Path(__file__).parent.parent.parent
CONFIG_FILE = Path(BASE_PATH, 'sgg_configs/vgattr/vinvl_x152c4.yaml')

MODEL_DIR = Path(BASE_PATH, "models/vinvl_vg_x152c4")
_MODEL_URL = "https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth"
_LABEL_URL = "https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/VG-SGG-dicts-vgoi6-clipped.json"


class VinVLVisualBackbone(object):
    def __init__(self, config_file=None, opts=None):

        num_of_gpus = torch.cuda.device_count()
        set_seed(1000, num_of_gpus)
        self.device = cfg.MODEL.DEVICE

        self.opts = {
            "MODEL.WEIGHT": "models/vinvl_vg_x152c4/vinvl_vg_x152c4.pth",
            "MODEL.ROI_HEADS.NMS_FILTER": 1,
            "MODEL.ROI_HEADS.SCORE_THRESH": 0.2,
            "TEST.IGNORE_BOX_REGRESSION": False,
            "DATASETS.LABELMAP_FILE": "models/vinvl_vg_x152c4/VG-SGG-dicts-vgoi6-clipped.json",
            "TEST.OUTPUT_FEATURE": True
        }
        if opts:
            self.opts.update(opts)

        if config_file:
            self.config_file = config_file
        else:
            #    raise ValueError("You need to pass a config_file")
            self.config_file = CONFIG_FILE

        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.merge_from_file(self.config_file)
        cfg.update(self.opts)
        cfg.set_new_allowed(False)
        cfg.freeze()

        if cfg.MODEL.META_ARCHITECTURE == "AttrRCNN":
            self.model = AttrRCNN(cfg)
        else:
            raise ValueError(
                f"{cfg.MODEL.META_ARCHITECTURE} is not a valid MODEL.META_ARCHITECTURE; it must be 'AttrRCNN'")

        self.model.eval()
        self.model.to(self.device)

        if not Path(Path(BASE_PATH, cfg.MODEL.WEIGHT)).is_file():
            print(f"{cfg.MODEL.WEIGHT} not found")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            print(f"created {MODEL_DIR} ")


            print(f"downloading {Path(_MODEL_URL).name}")
            # download the model
            r = requests.get(_MODEL_URL, stream=True)
            path = Path(MODEL_DIR, Path(_MODEL_URL).name)
            print(f"downloading {Path(_MODEL_URL).name} in {path}")
            with open(path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                    if chunk:
                        f.write(chunk)
                        f.flush()

            # dowload the labelmap
            print(f"downloading {Path(_LABEL_URL).name}")
            r = requests.get(_LABEL_URL, stream=True)
            path = Path(MODEL_DIR, Path(_LABEL_URL).name)
            with open(path, 'wb') as f:
                total_length = int(r.headers.get('content-length'))
                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                    if chunk:
                        f.write(chunk)
                        f.flush()

        self.checkpointer = DetectronCheckpointer(cfg, self.model, save_dir="")
        self.checkpointer.load(str(Path(BASE_PATH, cfg.MODEL.WEIGHT)))

        with open(Path(BASE_PATH, cfg.DATASETS.LABELMAP_FILE), "rb") as fp:
            label_dict = json.load(fp)

        self.idx2label = {int(k): v for k, v in label_dict["idx_to_label"].items()}
        self.label2idx = {k: int(v) for k, v in label_dict["label_to_idx"].items()}

        self.transforms = build_transforms(cfg, is_train=False)

    def __call__(self, img):

        if isinstance(img, str):
            img = cv2.imread(img)

        # else assume a cv2.imread
        # cv2_img is the original input, so we can get the height and
        # width information to scale the output boxes.
        img_input = cv2Img_to_Image(img)
        img_input, _ = self.transforms(img_input, target=None)
        img_input = img_input.to(self.model.device)

        with torch.no_grad():
            prediction = self.model(img_input)
            prediction = prediction[0].to(torch.device("cpu"))

        img_height = img.shape[0]
        img_width = img.shape[1]

        prediction = prediction.resize((img_width, img_height))
        boxes = prediction.bbox.tolist()
        classes = [self.idx2label[c] for c in prediction.get_field("labels").tolist()]
        scores = prediction.get_field("scores").tolist()
        features = prediction.get_field("box_features").cpu().numpy()
        spatial_features = encode_spatial_features(features, (img_width, img_height), mode="xyxy")

        return {
            "boxes": boxes,
            "classes": classes,
            "scores": scores,
            "features": features,
            "spatial_features": spatial_features
        }
