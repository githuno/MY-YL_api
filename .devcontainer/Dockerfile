FROM python:3.9-buster
ENV PYTHONUNBUFFERED=1

COPY app/ /src/
WORKDIR /app

RUN pip3 install --upgrade pip
# RUN pip3 install fastapi
# RUN pip3 install "uvicorn[standard]"
# RUN pip3 install requests

RUN pip3 install fastapi
RUN pip3 install uvicorn[standard]
RUN pip3 install pillow
RUN pip3 install numpy
RUN pip3 install tensorflow
RUN pip3 install python-multipart

RUN pip3 install Jinja2
RUN pip3 install aiofiles
RUN pip3 install opencv-python

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN git clone --depth=1 https://github.com/githuno/yolov5.git
RUN pip3 install -qr yolov5/requirements.txt
RUN mv test_predict.py yolov5/classify/

# ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]

# コンテナの起動時に実行するコマンドを指定
# CMD ["python", "main.py"]