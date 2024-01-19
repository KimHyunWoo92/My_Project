import streamlit as st
from st_circular_progress import CircularProgress
import requests
import tempfile
import cv2
import json



st.set_page_config(layout="wide",page_title='FAKEID',page_icon='web/assets/face_id.png')
st.title('FAKE ID')
empty1, con1, empty2 = st.columns([1,8,1])

video=None
output=None

def model_predict(data):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(data.read())
    video = open(tfile.name,'rb')
    
    file = {
        'file' : video
    }
    
    url = 'http://127.0.0.1:8080/files/'
    print('requests post go!')
    x = requests.post(url=url,files=file)
    response = json.loads(x.text)
    output = response['output']
    result = response['result']
    print(output,result)
    return output,result

def change_progress(result):
    color= 'red' if int(result) > 50 else'green'
    
    my_circular_progress.update_value(int(result))
    my_circular_progress.color=color
    
my_circular_progress=CircularProgress(
            label='Fake or Real',
            value=0,
            color="white",
            size='large')

with empty1:
    st.empty()

with con1:
    st.session_state['video']=st.file_uploader('VIDEO',type=['mp4'])
    video=st.session_state['video']

    col1, col2 = st.columns([7,3])
    with col2:
        result=0
        if video:
            if st.button('분석하기',use_container_width=True):
                output,result = model_predict(video)
                change_progress(result)
        else:
            st.warning("분석할 영상을 입력해주세요")
        # my_circular_progress.st_circular_progress()
        if result:
            st.header(f'분석결과 : {"Fake" if result > 50 else "Real"}')
        else:
            st.header("분석결과 대기중..")
        my_circular_progress.st_circular_progress()

        
    with col1:
        if (video is not None) and (output is None):
            data=st.video(video)
            # print(type(data))
        elif output is not None:
            # print(output)
            st.video(output)
        else:
            st.video('https://youtu.be/bsrrCyn5L5I?si=ntDF2WQZBqvwev5H')
            


    
        
        
with empty2:
    st.empty()