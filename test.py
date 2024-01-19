import torch
import torch.nn as nn
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
import mydata
import extract_face
import shutil
import cv2
import os

def test(video_path,model_path):
    extract_face.extract_img(video_path)
    images=mydata.read_data()
    dataset = mydata.MyDataset(file_list=images, transform=xception_default_data_transforms['test'])
    batch_size=32
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,num_workers=8)
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()
    fakes = 0
    preds_frames=[]
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1
    with torch.no_grad():
        for image in data_loader:
            image = image.cuda()

            outputs=model(image)
            _, preds = torch.max(outputs.data, 1)
            preds_frames += preds.tolist()
            fakes += torch.sum(preds == 0).to(torch.float32)
        print(f'Fakes: {int(fakes)}   Length: {len(dataset)}   Fake: {round(float(fakes/len(dataset)) * 100, 2)}%')
        persent_output=round(float(fakes/len(dataset)) * 100, 2)
    image_list=os.listdir('./Extract_Imgs')
    for image_name, pred in zip(image_list,preds_frames):
        image_path = os.path.join('./Extract_Imgs',image_name)
        image=cv2.imread(image_path)
        if image is not None:
            if pred==1:
                cv2.rectangle(image, (0, 0), (image.shape[0],image.shape[1]), color=(0,255,0), thickness=2)
                cv2.putText(image,'Real',(0,30),font_face,font_scale,(0,255,0),thickness,2)
            else:
                cv2.rectangle(image, (0, 0), (image.shape[0],image.shape[1]), color=(0,0,255), thickness=2)
                cv2.putText(image,'Fake',(15,15),font_face,font_scale,(0,0,255),thickness,2)
            cv2.imshow('Output',image)
            cv2.waitKey(10)
    cv2.destroyAllWindows()
    shutil.rmtree('./Extract_Imgs')
    return persent_output

if __name__== '__main__':
    persent_output=test(video_path='https://storage.googleapis.com/deep_fake_dataset/input/6537b658-b180-4d3b-9c82-d185675b74b3.mp4',model_path="./models/11_deepburger.pkl")
    print(persent_output)