import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def preprocess(image_path) :
    # 画像を読み込む
    img = Image.open(image_path)

    # 画像をNumPy配列に変換する
    img_arr = np.array(img)

    # NumPy配列を正規化する
    img_arr = img_arr.astype(np.float32) / 255.0
    img_arr = (img_arr - 0.5) / 0.5

    # NumPy配列をトーチテンソルに変換する
    img_tensor = torch.from_numpy(img_arr)

    # テンソルの形状を変更する
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW形式に変更する

    # テンソルを表示する
    print(img_tensor.shape)

    return img_tensor

def predict(image_tensor) :
    # validate
    if image_tensor is None:
        raise ValueError("Input image is None")
    
    # load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s-cls.pt')
    names = model.names

    results = model(image_tensor)
    pred = F.softmax(results, dim=1)  # probabilities
    top5i = pred[0].argsort(0, descending=True)[:5].tolist() # top 5 indices

    response = []
    for i in top5i:
        res = {}
        res["labels"] = names[i]
        res["confidence"] = f'{pred[0][i]:.2f}'
        response.append(res)

        response

    print(f'結果：{response}')
    return response