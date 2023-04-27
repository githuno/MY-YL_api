from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import os
import shutil
import predict as pdt

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post('/api/predict')
async def predict_image(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return JSONResponse(content={"error": "Image must be jpg or png format!"})
    
    # 一時ファイルのパスを取得する
    file_path = os.path.join("/tmp", file.filename)
    
    # 受け取ったファイルを一時ファイルに保存する
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 一時ファイルのパスをpredict関数に渡す
    image = await pdt.preprocess(file_path)
    pred = await pdt.predict(image)
    # 一時ファイルを削除する
    os.remove(file_path)
    
    response = {"prediction": pred}
    return JSONResponse(content=jsonable_encoder(response))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8080, host='0.0.0.0')
