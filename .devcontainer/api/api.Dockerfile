FROM python:3.9-buster
ENV PYTHONUNBUFFERED=1

WORKDIR /app
# 元ディレクトリapi配下のデータを、app/api配下に
# （コンテナapiディレクトリはこの時点で新規に作成）
COPY api/ /app/api
# その後、作業ディレクトリを移動
WORKDIR /app/api

RUN pip3 install --upgrade pip
# RUN pip3 install requests

RUN pip3 install fastapi
RUN pip3 install uvicorn[standard]
RUN pip3 install pillow
RUN pip3 install numpy
RUN pip3 install --default-timeout=1000 tensorflow
RUN pip3 install python-multipart

RUN pip3 install Jinja2
RUN pip3 install aiofiles
RUN pip3 install opencv-python

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN git clone --depth=1 https://github.com/githuno/yolov5.git
RUN pip3 install --default-timeout=1000 -r yolov5/requirements.txt
RUN mv test_predict.py yolov5/classify/

# ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]

# コンテナの起動時に実行するコマンドを指定
# CMD ["python", "main.py"]
ENTRYPOINT ["python", "main.py"]