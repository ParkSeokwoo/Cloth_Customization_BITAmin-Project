# BITAmin Project #01. Recommendation System

- 분야: 추천시스템, CV
- 주제: 입력한 옷과 TPO를 고려한 의류추천
- 참여조원: 배성윤, 서은서, 황석우, 김재겸

---

### INTRO
👚 오늘 뭐입지?! 👕

💬 : 🚨 설마 너 지금.. 그렇게 입고 나가게? 🚨

패션센스가 2% 부족한 당신을 위해 준비했습니다!
사진 이미지만 입력하면, 요즘 트렌디한 스타일과 여러분의 TPO를 고려하여 코디를 추천해드립니다. 
무신사와 온더룩의 패셔니스타들의 코디를 지금 바로 참고해보세요!

![fashionista](https://github.com/baesunny/cloth_customization/assets/133308712/7032a36c-105f-404a-8a3f-6cfde1973f0b)
(이미지 출처: https://onthelook.co.kr/)

---

### PROCESS

1. 온더룩, 무신사 사이트에서 데이터 크롤링 (상품, 스타일)
2. 이미지 Segmentation (Pre-trained 모델 4가지 중 최종 선정모델: Segformer)
3. Object Detection & Cropping
4. 전처리 완료된 이미지 벡터화
5. 파일명 통일 후 폴더별 저장
6. 이미지 유사도 분석
7. streamlit 앱 구현


<img width="1000" alt="process" src="https://github.com/baesunny/cloth_customization/assets/133308712/b1b4a776-ac34-422a-b317-fdbf45a6d553">

---

### RESULT

결과물은 streamlit 앱으로 구현하였으며, 발표자료와 구현영상과 은 아래에 첨부해두었다.

- 발표자료
[오늘 뭐입지.pdf](https://github.com/baesunny/cloth_customization/files/14444764/default.pdf)

- 구현영상
https://github.com/baesunny/cloth_customization/assets/133308712/0329369e-e4b7-4f10-bdeb-af3c13818302


---

