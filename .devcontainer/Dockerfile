FROM python:3.9-buster AS builder
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# RUN apt-get update 
# RUN apt-get install -y libbzip2-dev zlib1g-dev libpng-dev libharfbuzz-dev libfreetype6-dev

RUN pip3 install --upgrade pip
RUN pip3 install fastapi
RUN pip3 install uvicorn[standard]
RUN pip3 install pillow>=7.1.2
RUN pip3 install matplotlib>=3.3
RUN pip3 install seaborn>=0.11.0
RUN pip3 install numpy>=1.18.5
RUN pip3 install pandas>=1.1.4
RUN pip3 install requests>=2.23.0
RUN pip3 install psutil
RUN pip3 install torch==1.7.1
RUN pip3 install torchvision==0.8.2
RUN pip3 install tqdm>=4.64.0
RUN pip3 install python-multipart
RUN pip3 install opencv-python

# -----------------------------------------------------
# https://qiita.com/nokonoko_1203/items/204da4a9d5247c276136
# -----------------------------------------------------
FROM python:3.9-slim-buster AS production
ENV PYTHONUNBUFFERED=1

WORKDIR /app
# 元ディレクトリapi配下のデータを、app/api配下に
# （コンテナapiディレクトリはこの時点で新規に作成）
COPY api_s/ /app/api
# その後、作業ディレクトリを移動
WORKDIR /app/api

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages/

RUN apt-get update 
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 libjpeg-dev libopenjp2-7 libtiff5-dev
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENTRYPOINT ["python", "main.py", "--host", "0.0.0.0", "--port", "8080" ]