import os
import shutil
from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
import predict as pdt

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    # 一時ファイルのパスを取得する
    file_path = os.path.join("/tmp", file.filename)
    
    # 受け取ったファイルを一時ファイルに保存する
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 一時ファイルのパスをpredict関数に渡す
    pred = pdt.predict(file_path)
    # 一時ファイルを削除する
    os.remove(file_path)
    return pred

if __name__ == "__main__":
    # uvicorn.run(app, port=8080, host='0.0.0.0')
    uvicorn.run(app, host='0.0.0.0')

# from fastapi import Depends, FastAPI, HTTPException, Security
# from fastapi.security.api_key import APIKeyHeader, APIKey
# from starlette.status import HTTP_403_FORBIDDEN

# from routers import test
# from util import util

# correct_key: str = util.get_apikey()
# api_key_header = APIKeyHeader(name='Authorization', auto_error=False)

# async def get_api_key(
#     api_key_header: str = Security(api_key_header),
#     ):
#     if api_key_header == correct_key:
#         return api_key_header
#     else:
#         raise HTTPException(
#             status_code=HTTP_403_FORBIDDEN, detail="Could not validate credentials"
#         )

# app = FastAPI()
# app.include_router(test.router, dependencies=[Depends(get_api_key)], tags=["Test"])

# @app.get("/")
# def read_root():
#     return {"status": "ok"}