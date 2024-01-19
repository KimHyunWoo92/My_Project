from typing import Annotated
import uuid
import os
import shutil

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.google_cloud.storage_c import GCS
from src.google_cloud.firebase_c import GCFS
from test_from_video import test_from_video


gcs=GCS()
fb=GCFS()

app = FastAPI()

@app.get("/")
def read_root():


    content = """
<body>
    <form action="/files/" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
    </form>
    <form action="/uploadfiles/" enctype="multipart/form-data" method="post">
        <input name="files" type="file" multiple>
        <input type="submit">
    </form>
</body>
    """
    return HTMLResponse(content=content)

# dataset = { 
#     datapath
# }
class Item(BaseModel):
    name : str
    file : str


@app.post("/files/")
async def create_files(file: bytes = File()):
    print(type(file))
    UPLOAD_DIR = "./downloader"
    filename=f"{str(uuid.uuid4())}.mp4"

    # storage 저장
    # print(type(file)==type(bytes()))
    input_url=gcs.upload_to_bucket("input/",filename,file)

    # model_path=""
    # output_path="./downloader/output"
    
    # predict=67.08
    # output,predict=test_full_image_network(url,model_path,output_path,start_frame=0,end_frame=None,cuda=True)
    model_path="./models/11_deepburger.pkl"
    output,result =test_from_video(video_path=input_url)
    
    # print(type(output)==type(str()))
    output_url=gcs.upload_to_bucket("output/",filename,output)


    fb.insert_data(input_url,output_url,result)
    
    if os.path.isfile(output):
        os.remove(output)
    # with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
    #     fp.write(content)
    
    

    return {"output": output_url, "result":result}


    # return dataset