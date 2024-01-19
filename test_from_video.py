import torch
import torch.nn as nn
from network.models import model_selection
from dataset.transform import xception_default_data_transforms
import cv2
import os
from face_detector import YoloDetector
from PIL import Image as pil_image

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def preprocess_image(image, cuda=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))

    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    preprocessed_image = preprocess_image(image, cuda)

    output = model(preprocessed_image)
    output = post_function(output)

    _, prediction = torch.max(output, 1)
    prediction = float(prediction.cpu().numpy())

    return int(prediction), output


def test_from_video(video_path, cuda=True):
    reader = cv2.VideoCapture(video_path)
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_fn = video_path.split('/')[-1].split('.')[0]+'.mp4'
    writer = None

    face_detector = YoloDetector()

    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model.load_state_dict(torch.load("./models/11_deepburger.pkl"))
    model.eval()

    if cuda:
        model = model.cuda()

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    face_count=0
    fake_count=0
    frame_num = 0

    while reader.isOpened():
        _, image = reader.read()

        if image is None:
            break
            
        frame_num += 1

        height, width = image.shape[:2]
        bboxes,_=face_detector.predict(image)

        if writer is None:
            writer = cv2.VideoWriter(os.path.join('./downloader', video_fn), 0x00000021, fps,
                                     (height, width)[::-1])
            
        if len(bboxes[0]):
            box=bboxes[0][0]
            x, y, size = get_boundingbox(box, width, height)
            cropped_face = image[y:y+size, x:x+size]
            preprocessed_image = preprocess_image(cropped_face, cuda)
            outputs=model(preprocessed_image)
            _, preds = torch.max(outputs.data, 1)
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            label = 'fake' if preds == 0 else 'real'
            face_count+=1
            color = (0, 255, 0) if preds == 1 else (0, 0, 255)
            if label == 'fake':
                cv2.putText(image,'Fake', (x,y-10), font_face, font_scale, color, thickness)
                fake_count += 1
            else:
                cv2.putText(image,'Real', (x,y-10), font_face, font_scale, color, thickness)
            cv2.rectangle(image, (x, y), (w,h), color, 2)
        if face_count != 0:
            fake_percentage= round((fake_count/face_count)*100,2)
            fake_color = (0, 255, 0) if fake_percentage<=20.0 else (0, 0, 255)
            cv2.putText(image,'Fake='+str(fake_percentage)+'%',(10,30),
                            font_face,font_scale,
                            fake_color,thickness,2)
                
        if frame_num >= num_frames:
            break

        cv2.imshow('test', image)
        cv2.waitKey(33)
        writer.write(image)
        
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
        
    return os.path.join('./downloader', video_fn),fake_percentage

if __name__== '__main__':
    output,result=test_from_video(video_path='./videos/jisoo.mp4')
    print(output,result)