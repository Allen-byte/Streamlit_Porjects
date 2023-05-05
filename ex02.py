# coding: utf-8

import streamlit as st
from PIL import Image
import cv2 as cv
from aip import AipFace
from aip import AipImageClassify
import base64, io
import numpy as np

# 百度人脸识别 APPID
APP_ID = '25722045'
API_KEY = 'xBncwPok33dZzSOhjLRf5EG3'
SECRET_KEY = 'pgrjjT6GbEZDnsUilUN4xOaTgk6UDGNF'

# 百度图像识别参数
APP_ID_1 = '33185622'
API_KEY_1 = 'GKV7qrz8E2HIYDBbUldXFuu9'
SECRET_KEY_1 = 'riBadkDBaA5e23xQzKGdIzAdFSYOrDdW'


client = AipFace(APP_ID, API_KEY, SECRET_KEY)
client_1 = AipImageClassify(APP_ID_1, API_KEY_1, SECRET_KEY_1)


# 面容检测
def face(img_data, client):
    data = base64.b64encode(img_data)
    image = data.decode()
    imageType = "BASE64"
    # client.detect(image, imageType)

    options = {"face_field": "beauty,age,face_shape,expression,gender", "max_face_num": 10,"face_type":"LIVE"}
    result = client.detect(image, imageType,options)

    age = result['result']['face_list'][0]['age']
    faceValue = result['result']['face_list'][0]['beauty']
    return age, faceValue


# 传入的参数是Image图片
def face_value(client, image):
    # 将Image对象转为cv对象
    img = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("D:/pythonWorkSpace/static/haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray_img, 1.1, 5)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
    # 将cv对象转为bytes
    img_bytes = cv.imencode('.jpg', img)[1]
    age, faceValue = face(img_bytes, client)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return cv.cvtColor(img, cv.COLOR_BGR2RGB), age, faceValue


# 动物识别
def animal_detection(client, image):
    options = {}
    options['baike_num'] = 5
    res = client.animalDetect(image, options)
    max_score, name = 0.0, ''
    desc = ''
    for result in res['result']:
        score = round(float(result["score"]), 2)
        if score >= max_score:
            max_score = score
            name = result["name"]
            desc = result['baike_info']['description']
    return name, desc


# 菜品识别
def dish_detection(client, image):
    options = {}
    options["filter_threshold"] = "0.7"
    options["baike_num"] = 5
    res = client.dishDetect(image, options)
    name, max_score, calorie = '', 0, 0
    for result in res['result']:
        if result['has_calorie']:
            score = round(float(result['probability']), 2)
            if score >= max_score:
                max_score = score
                name = result['name']
                calorie = result['calorie']
    return name, calorie


# Image对象转二进制流
def Image2Bytes(image):
    img_byte = io.BytesIO()
    image.save(img_byte, format='JPEG')
    return img_byte.getvalue()


uploaded_file = st.file_uploader('please choose the upload image')
tab1, tab2, tab3 = st.tabs(['面容检测', '动物识别', '菜品识别'])

if uploaded_file:
    image = Image.open(uploaded_file)
    with tab1:
        if st.button('开始测试', key=0):
            with st.spinner('please waiting...'):
                try:
                    img, age, faceValue = face_value(client, image)
                    st.image(img)
                    st.header('预测结果如下：')
                    if faceValue > 80.00:
                        text = f'颜值: {faceValue}(惊为天人!)'
                        st.balloons()
                    else:
                        text = f'颜值: {faceValue}'
                    st.write(f'年龄: {age}')
                    st.write(text)
                except:
                    st.warning('请上传正确类型的图片或稍后再试')
    with tab2:
        if st.button('开始识别', key=1):
            with st.spinner('please waiting...'):
                try:
                    bytes_image = Image2Bytes(image)
                    name, desc = animal_detection(client_1, bytes_image)
                    st.image(image)
                    st.header('识别结果如下: ')
                    st.write(f'名称: {name}')
                    st.write(desc)
                except:
                    st.warning('请上传正确类型的图片或稍后再试')
    with tab3:
        if st.button('开始识别', key=2):
            with st.spinner('please waiting...'):
                try:
                    bytes_image = Image2Bytes(image)
                    name, calorie = dish_detection(client_1, bytes_image)
                    st.image(image)
                    st.header('识别结果如下: ')
                    st.write(f'名称: {name}')
                    st.write(f'热量: {calorie}')
                except:
                    st.warning('请上传正确类型的图片或稍后再试')
else:
    st.stop()