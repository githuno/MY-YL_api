# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

import argparse
from io import BytesIO
from PIL import Image
import os
import platform
import sys
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms, letterbox
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, print_args, strip_optimizer)
from utils.plots import Annotator
from utils.torch_utils import select_device, smart_inference_mode

def read_image(file_path: str) -> Image.Image:
    with open(file_path, 'rb') as f:
        image_encoded = f.read()
    pil_image = Image.open(BytesIO(image_encoded))
    return pil_image

def preprocess(image: Image.Image):
    image = np.asarray(image.resize((224, 224)))[..., :3]
    image = np.expand_dims(image, 0)
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image / 127.5 - 1.0

    return image

def LoadSingleImage(image, img_size=640, stride=32, auto=True, transforms=None):
    im0 = cv2.imread(image)  # BGR
    im = transforms(im0)  # transforms
    # if transforms:
    #     im = transforms(im0)  # transforms
    # else:
    #     im = letterbox(im0, img_size, stride=stride, auto=auto)[0]  # padded resize
    #     im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    #     im = np.ascontiguousarray(im)  # contiguous
    return im

# @smart_inference_mode()
def run(source=ROOT / 'data/images'):
    print(f"source: {source}, type: {type(source)}")

    # validate
    if source is None:
        raise ValueError("Input image is None")
    
    # Load model
    device = select_device('')
    model = DetectMultiBackend(ROOT / 'yolov5s-cls.pt', device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((224, 224), s=stride)  # check image size

    # sourceの中身を読み込みimageに入れる
    # image = read_image(source)
    # # image = preprocess(image)
    # print(f'typeofimage:{type(image)}')
    image = source
    
    # Dataloader
    # bs = 1  # batch_size
    # dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=1)
    # for path, im, im0s, vid_cap, s in dataset:
    #     print()
    # vid_path, vid_writer = [None] * bs, [None] * bs
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

    # print (f'dataset:{dataset["im"]}')
    # print(f'path:  {path}')
    # print(f'im:  {im}')
    # print(f'imos:  {im0s}')
    # print(f'vid_cam:  {vid_cap}')
    # print(f's:  { s}')
    print(f'dt[0]:  {dt[0]}')
    print(f'dt[1]:  {dt[1]}')
    print(f'dt[2]:  {dt[2]}')
    # Process predictions
    top5i = pred[0].argsort(0, descending=True)[:5].tolist()  # top 5 indices

    # 参考：https://techpr.info/ml/fastapi-tensorflow-imager-recognition/
    response = []
    for j in top5i:
        resp = {}
        resp["class"] = names[j]
        resp["confidence"] = f'{pred[0][j]:.2f}'

        response.append(resp)

    print(f'結果：{response}')
    return response    

def main():
    # check_requirements(exclude=('tensorboard', 'thop'))
    parser = argparse.ArgumentParser(description='Image classification with YOLOv5')
    parser.add_argument('image_path', type=str, help='Path to input image')
    args = parser.parse_args()

    run(source=args.image_path)

if __name__ == '__main__':
    # opt = parse_opt()
    main()
    print('start')

# def read_image(file_path: str) -> Image.Image:
#     with open(file_path, 'rb') as f:
#         image_encoded = f.read()
#     pil_image = Image.open(BytesIO(image_encoded))
#     return pil_image


# def preprocess(image: Image.Image):
#     image = np.asarray(image.resize((224, 224)))[..., :3]
#     image = np.expand_dims(image, 0)
#     if image.dtype == np.float64:
#         image = (image * 255).astype(np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     image = image / 127.5 - 1.0

#     return image

# def run(image: np.ndarray) :
#     # Preprocess
#     if image is None:
#         raise ValueError("Input image is None")
    
#     device = select_device('')
#     model = DetectMultiBackend(ROOT / 'yolov5s-cls.pt', device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size((224, 224), s=stride)  # check image size

#     im = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     im = torch.Tensor(im).to(model.device)
#     im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#     if len(im.shape) == 3:
#         im = im[None]  # expand for batch dim

#     # Inference
#     results = model(im)

#     # Post-process
#     pred = F.softmax(results, dim=1)  # probabilities

#     # Process predictions
#     top5i = pred[0].argsort(0, descending=True)[:5].tolist()  # top 5 indices

#     response = []
#     for j in top5i:
#         resp = {}
#         resp["class"] = names[j]
#         resp["confidence"] = f'{pred[0][j]:.2f}'

#         response.append(resp)

#     print(f'結果：{response}')
#     return response

# def main():
#     parser = argparse.ArgumentParser(description='PyTorch Image Classification')
#     parser.add_argument('--image', type=str, default='dog.jpeg', help='input image')
#     args = parser.parse_args()

#     image = read_image(args.image)
#     image = preprocess(image)
#     run(image)

# if __name__ == '__main__':
#     main()
#     print('start')
