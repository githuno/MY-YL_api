# import argparse
import os
# import platform
import sys
from pathlib import Path
# from tkinter import image_types
# from PIL import Image
# from io import BytesIO
import cv2
import numpy as np

ROOT = Path('/app/api/yolov5')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# https://note.nkmk.me/python-import-module-search-path/

import torch
import torch.nn.functional as F
from utils.augmentations import classify_transforms, letterbox
# from utils.dataloaders import LoadImages

from utils.general import Profile, check_img_size


from models.common import DetectMultiBackend
# from utils.general import check_img_size
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, print_args, strip_optimizer)
# from utils.plots import Annotator
from utils.torch_utils import select_device #, smart_inference_mode


# import io
# from typing import List

# import tensorflow as tf
# # from keras.applications import resnet
# from fastapi import FastAPI, Request, File, UploadFile
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import JSONResponse

# def read_image(image_encoded: bytes):
#     pil_image = Image.open(BytesIO(image_encoded))
#     return pil_image if pil_image.mode == "RGB" else pil_image.convert("RGB")

# def read_image(image_encoded: bytes):
#     pil_image = Image.open(BytesIO(image_encoded))

#     return pil_image # PILのイメージオブジェクト

# def preprocess(image: Image.Image):
#     image = np.asarray(image.resize((224, 224)))[..., :3] # numpy.ndarray型に変更
#     image = np.expand_dims(image, 0)
#     image = image / 127.5 - 1.0

#     return image

def LoadSingleImage(source, img_size=640, stride=32, auto=True, transforms=None):
    im0 = cv2.imread(str(source))  # BGR
    # im = transforms(im0)  # transforms
    if transforms:
        im = transforms(im0)  # transforms
    else:
        im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
    return im

# @smart_inference_mode()
def predict(image) :
    # preprocess
    if image is None:
        raise ValueError("Input image is None")
    
    device = select_device('')
    model = DetectMultiBackend(ROOT / 'yolov5s-cls.pt', device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((224, 224), s=stride)  # check image size

    im = LoadSingleImage(image, img_size=imgsz, stride=stride, auto=True, transforms=classify_transforms(imgsz[0]))

    dt = (Profile(), Profile(), Profile())
    with dt[0]:
        im = torch.Tensor(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        results = model(im)

    # Post-process
    with dt[2]:
        pred = F.softmax(results, dim=1)  # probabilities

    # Process predictions
    top5i = pred[0].argsort(0, descending=True)[:5].tolist()
    response = []
    for j in top5i:
        resp = {}
        resp["class"] = names[j]
        resp["confidence"] = f'{pred[0][j]:.2f}'

        response.append(resp)

    print(f'結果：{response}')
    return response

    # # Load model
    # device = select_device('')
    # model = DetectMultiBackend(ROOT / 'yolov5s-cls.pt', device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)
    # names, pt = model.names, model.pt

    # # Inference
    # model.warmup(imgsz=(1 if pt else 1, 3, [224, 224]))  # warmup
    # im = torch.from_numpy(image.copy()).to(model.device)
    # im = im.half() if model.fp16 else im.float()
    # if len(im.shape) == 3:
    #     im = im[None]  # expand for batch dim
    # results = model(im)
    # return results
    # pred = F.softmax(results, dim=1)  # probabilities

    # # Process predictions
    # response = []
    # top5i = pred.argsort(0, descending=True)[:5].tolist()  # top 5 indices
    # for j in top5i:
    #     resp = {}
    #     resp["class"] = names[j]
    #     resp["confidence"] = f'{pred[j]:.2f}'
    #     response.append(resp)

    # return response