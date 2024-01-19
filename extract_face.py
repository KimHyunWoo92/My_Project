import cv2
from face_detector import YoloDetector
import os

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

def extract_img(video_path):
    os.mkdir(f'./Extract_Imgs')
    reader = cv2.VideoCapture(video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    face_detector = YoloDetector()
    frame_count = 0
    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break

        height, width = image.shape[:2]
        bboxes,_=face_detector.predict(image)

        if len(bboxes[0]):
            box=bboxes[0][0]
            x, y, size = get_boundingbox(box, width, height)
            cropped_face = image[y:y+size, x:x+size]
            cv2.imwrite(f"./Extract_Imgs/{'0'*(6-len(str(frame_count)))+str(frame_count)}.jpg",cropped_face)
            frame_count+=1

        if frame_count >=num_frames:
            break

if __name__ == '__main__':
    extract_img(video_path='C:\\Users\\administrator\\Desktop\\train-model\\videos\\fake04.mp4')
