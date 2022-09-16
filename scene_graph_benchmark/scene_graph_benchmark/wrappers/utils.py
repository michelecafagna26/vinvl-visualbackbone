import cv2
from PIL import Image
import numpy as np


def encode_spatial_features(bbox, img_size, mode="xyxy"):
    img_w, img_h = img_size
    # all the boxes
    if mode == "xyxy":

        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        bw = x2 - x1
        bh = y2 - y1

    elif mode == "xywh":
        x1, y1, bw, bh = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    else:
        raise ValueError(f"mode uknown: {mode}; it can be either 'xyxy' or 'xywh'")

    scaled_x = x1 / img_w
    scaled_y = y1 / img_h
    scaled_w = bw / img_w
    scaled_h = bh / img_h

    # dim (num_feat, num_boxes)
    return np.stack([scaled_x, scaled_y, scaled_x + scaled_w, scaled_y + scaled_h, scaled_w, scaled_h], axis=1)


def decode_spatial_features(spatial, img_size, mode="xywh"):
    # all the boxes
    # spatial = [scaled_x, scaled_y, scaled_x + scaled_w, scaled_y + scaled_h, scaled_w, scaled_h]

    if mode not in {"xyxy", "xywh"}:
        raise ValueError(f"mode uknown: {mode}; it can be either 'xyxy' or 'xywh'")

    img_w, img_h = img_size
    scaled_x, scaled_y = spatial[:, 0], spatial[:, 1]
    scaled_w, scaled_h = spatial[:, 2] - scaled_x, spatial[:, 3] - scaled_y
    scaled_w, scaled_h = spatial[:, 4], spatial[:, 5]

    x1, y1, bw, bh = scaled_x * img_w, scaled_y * img_h, scaled_w * img_w, scaled_h * img_h
    bbox = [x1, y1, bw, bh]

    if mode == "xyxy":
        x2 = x1 + bw
        y2 = y1 + bh
        bbox = [x1, y1, x2, y2]

    return np.stack(bbox, axis=1)


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img
