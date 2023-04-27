import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# from torchvision import transforms # 追加(chatGPT案)

# ロードをグローバルスコープに移動し、アプリケーションの起動時に1回だけ実行されるように
# リクエストごとにモデルをロードすることを回避し、メモリ使用量を削減
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s-cls.pt')
names = model.names

async def preprocess(image_path, max_size=(640, 640)) :
    # 画像を読み込む
    img = Image.open(image_path)
    img.thumbnail(max_size) # アスペクト比を維持しながら画像サイズを縮小
    img_arr = np.array(img) # 画像をNumPy配列に変換する
    img.close()  # メモリリーク予防で画像はクローズ

    # NumPy配列を正規化する
    img_arr = img_arr.astype(np.float32) / 255.0 # 浮動小数点数(float32)に変換し,255で割って0から1の範囲に変換
    img_arr = (img_arr - 0.5) / 0.5 # 各pixel値から0.5を引き([-0.5、0.5]の範囲に変換)、0.5で割る([-0.5、0.5]の範囲に変換)
    img_tensor = torch.from_numpy(img_arr) # NumPy配列をトーチテンソルに変換する
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # テンソルの形状をBCHW形式に変更する

    # # ここで画像のサイズを変更(chatGPT案)
    # resize = transforms.Resize((128, 128)) # サイズは適宜調整
    # img_tensor = resize(img_tensor)

    return img_tensor

async def predict(image_tensor) :
    # validate
    if image_tensor is None:
        raise ValueError("Input image is None")

    # load model → グローバルに移動し、毎回読み込まないように変更

    results = model(image_tensor)
    pred = F.softmax(results, dim=1)  # probabilities
    top5i = pred[0].argsort(0, descending=True)[:5].tolist() # top 5 indices

    response = []
    for i in top5i:
        res = {}
        res["labels"] = names[i]
        res["confidence"] = f'{pred[0][i]:.2f}'
        response.append(res)

    print(f'結果：{response}')
    return response

    # トーチテンソルとはPyTorchで扱われる、
    # 多次元配列を表現するためのデータ構造のことをトーチテンソル（torch.Tensor）と呼びます。
    # Numpyのndarrayとよく似ており、CPUやGPU上のデータを扱えるため、
    # 機械学習やディープラーニングのフレームワークで広く利用されています。
    # また、トーチテンソルは自動微分をサポートしており、
    # ニューラルネットワークの学習に必要な勾配計算を簡単に実現することができます。

    # TorchとNumPyは、両方とも数値計算や科学技術計算において頻繁に使用されるPythonのライブラリです。
    # 両方のライブラリは、数値計算に必要な多次元配列や行列演算を提供しています。
    # NumPyは、Pythonにおける数値計算の基本的なライブラリとして有名で、
    # 多次元配列オブジェクト（ndarray）を提供します。Torchも同様に、多次元テンソルを扱うことができます。
    # Torchは、NumPyと同様に、数値計算や科学技術計算のための高度な数学関数やアルゴリズムを提供しますが、
    # 深層学習のための高レベルのAPIやツールも提供します。また、TorchはGPUを利用した高速な計算が可能であり、
    # 深層学習のような大規模な計算でも高速に処理できます。したがって、TorchとNumPyは両方とも似たような処理を提供していますが、
    # Torchは主に深層学習のためのライブラリであり、GPUを利用して高速な計算が可能な点が異なります。