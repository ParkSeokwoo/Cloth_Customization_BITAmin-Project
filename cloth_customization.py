#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
from pylab import imshow
import matplotlib as mpl
import numpy as np
import torch
import cv2
import pandas as pd
import os

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from transformers import AutoImageProcessor, DetrForObjectDetection
# segmentation
processor_seg = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model_seg = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
#object detection
processor_obj = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model_obj = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")



### INTRO ###
st.header('👚 오늘 뭐입지?! 👕')
st.markdown('🚨 **설마 너 지금.. 그렇게 입고 나가게?** 🚨')
st.markdown(' **패션센스가 2% 부족한 당신을 위해 준비했습니다!** 사진 이미지만 입력하면, 요즘 트렌디한 스타일과 여러분의 TPO를 고려하여 코디를 추천해드립니다. 무신사와 온더룩의 패셔니스타들의 코디를 지금 바로 참고해보세요! ')
st.image('./intro_img/fashionista.jpg')

st.markdown('--------------------------------------------------------------------------------------')
st.header('PROCESS')
st.image('./intro_img/process.png')
st.markdown('--------------------------------------------------------------------------------------')



### INPUT ###
# # 의류 이미지 업로드
# input_image = st.file_uploader("의류 이미지를 업로드하세요. (배경이 깔끔한 사진이라면 더 좋습니다!)", type=['png', 'jpg', 'jpeg'])
# # 입력받은 의류 이미지 카테고리 선택
# input_cat = st.selectbox(
#     '귀하가 업로드한 의류 이미지의 카테고리를 골라주세요.',
#     ('top', 'bottom', 'shoes', 'hat', 'sunglasses', 'scarf', 'bag', 'belt'))
# st.write('You selected:', input)
# # 추천받고 싶은 의류 카테고리 선택
# output_cat = st.selectbox(
#     '추천받고 싶은 의류 카테고리를 선택해주세요.',
#     ('top', 'bottom', 'shoes', 'hat', 'sunglasses', 'scarf', 'belt'))
# if input == output:
#     st.error('Error: 업로드한 의류 카테고리와 다른 카테고리를 선택해주세요.')
# st.write('You selected:', output)
# # 상황 카테고리 선택
# situation = st.checkbox(
#     '상황 카테고리를 선택해주세요.',
#     ('여행', '카페', '전시회', '캠퍼스&출근', '급추위', '운동'))
# st.write('You selected:', situation)
# # 선택된 상황 카테고리를 영어로 변환해서 변수 저장
# situation_mapping = {
#     '여행': 'travel',
#     '카페': 'cafe',
#     '전시회': 'exhibit',
#     '캠퍼스&출근': 'campus_work',
#     '급추위': 'cold',
#     '운동': 'exercise'}
# situation = [situation_mapping[item] for item in situation]


# 임시로 변수 정의
input_image = './example_top.jpg'
input_cat = 'top'
output_cat = 'bottom'
situation = 'travel'



### 입력받은 이미지 segmentation & detection & vector변환 ###
image = Image.open(input_image)

# object detection & cropping 함수
def cropping(images,st = 1,
  fi = 0.0,
  step = -0.05):
  image_1 = Image.fromarray(images)
  inputs = processor_obj(images=image_1, return_tensors="pt")
  outputs = model_obj(**inputs)
  for tre in np.arange(st,fi,step):
    try:
        # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        target_sizes = torch.tensor([image_1.size[::-1]])
        results = processor_obj.post_process_object_detection(outputs, threshold=tre, target_sizes=target_sizes)[0]
        
        img = None
        for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            box = [round(i, 2) for i in box.tolist()]
            xmin, ymin, xmax, ymax = box
            img = image_1.crop((xmin, ymin, xmax, ymax))

        poss = np.array(img).sum().sum()
        return img
        break
    except:
        continue
  return images

# vector 변환 함수
default_path = './'
def image_to_vector(image,resize_size=(256,256)):  # 이미지 size 변환 resize(256,256)
    #image = Image.fromarray(image)
    #image = image.resize(resize_size)
    image = Image.fromarray(np.copy(image))
    image = image.resize(resize_size)
    image_array = np.array(image, dtype=np.float32)
    image_vector = image_array.flatten()
    return image_vector

# 전체 통합 함수
def final_image(image):    
    if len(np.array(image).shape) == 2:
        image = Image.fromarray(image).convert('RGB')
    # segmentation
    inputs = processor_seg(images=image, return_tensors="pt")
    outputs = model_seg(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    segments = torch.unique(pred_seg)
    default_path = './'
    
    for i in segments:
        if int(i) == 0:
            continue
        if int(i) == 1:
            cloth = 'hat'
            cloths = 'hat'
            mask = pred_seg == i
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 3:
            cloth= 'sunglasses'
            cloths= 'sunglasses'
            mask = pred_seg == i
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 4:
            cloth = 'top'
            cloths = 'top'
            mask = pred_seg == i
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) in [5,6,7]:
            cloth= ['pants','skirt','dress']
            cloths= 'bottom'
            mask  = (pred_seg == torch.tensor(5)) | (pred_seg == torch.tensor(6)) | (pred_seg == torch.tensor(7))
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 8:
            cloth = 'belt'
            cloths = 'belt'
            mask = pred_seg == torch.tensor(8)
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif (int(i) == 9):
            cloth = 'shoes'
            cloths = 'shoes'
            mask = (pred_seg == torch.tensor(9)) | (pred_seg == torch.tensor(10))
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 16:
            cloth = 'bag'
            cloths = 'bag'
            mask = pred_seg == torch.tensor(16)
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        elif int(i) == 17:
            cloth = 'scarf'
            cloths = 'scarf'
            mask = pred_seg == torch.tensor(17)
            image = np.array(image)
            mask_np = (mask * 255).numpy().astype(np.uint8)
            result = cv2.bitwise_and(image.astype(np.uint8), image.astype(np.uint8), mask=mask_np)
            img = cropping(result)
            img_vector = image_to_vector(img)
        return img_vector
    
# 입력받은 이미지 전처리 완료
input_img = final_image(image)




### 유사도 분석 ###
# 하나는 이미지, 다른 하나는 경로로 받는 경우
def cosine_similarity(vec1, vec2_path):
    vec2 = np.loadtxt(vec2_path)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)   
    return similarity
# 둘 다 경로로 받는 경우
def cosine_similarity_2(vec1_path, vec2_path):
    vec1 = np.loadtxt(vec1_path)
    vec2 = np.loadtxt(vec2_path)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

# 입력받은 이미지 & 동일 카테고리 폴더에 저장된 스타일 이미지
sim_list = []
file_path = './style/' + situation + '/' + input_cat + '/'   # ex) './cafe/top/'
cloths = os.listdir('./style/' + situation + '/' + input_cat + '/')
for cloth in cloths:
    sim_list.append(cosine_similarity(input_img, file_path + cloth))
max_idx = np.argmax(sim_list)
cloths[max_idx]
# target_image 정의
target_image = './style/' + situation + '/' + output_cat + '/' + cloths[max_idx]
# 유사도 분석 완료된 스타일seg 이미지와 product_seg 유사도분석
sim_list = []
file_path = './product/' + output_cat + '/'
cloths = os.listdir('./product/' + output_cat + '/')
for cloth in cloths:
    sim_list.append(cosine_similarity_2(target_image, file_path + cloth))
max_idx = np.argmax(sim_list)
cloths[max_idx]
## 예시 출력값: 'bottom_1883.txt'


# 최종 이미지 출력


# In[6]:


# input_image = './example_top.jpg'
# input_cat = 'top'
# output_cat = 'bottom'
# situation = 'travel'


# In[20]:


# # 해당 url 찾기
# with open('cafe_urls.txt', 'r') as file:
#     urls = file.readlines()
    
# urls[987]


# #### .txt 파일로 변환

# In[21]:


# file_names = os.listdir('../product/bottom_seg')
# file_path = '../product/bottom_seg'
# for name in file_names:
#     src = os.path.join(file_path, name)
#     dst = name + '.txt'
#     dst = os.path.join(file_path, dst)
#     os.rename(src, dst)


# #### 압축해제

# In[22]:


# # Open the zip file
# zip_file_path = '../product/bottom_seg-20240222T170613Z-001.zip'
# extract_dir = '../product'

# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     # Extract all the contents into the specified directory
#     zip_ref.extractall(extract_dir)

