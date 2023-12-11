# CARNIVAL_Automatic_Gain_Control
Carnival system을 구성하는 Automatic Gain Control 모델입니다.<br>
과학기술통신부 재원으로 정보통신기획평가원(IITP) 지원을 받아 수행한<br>
"원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제 공개 코드입니다.<br>
(2021.05~2024.12)<br>

2차년도 AGC 모델: Neural VAD와 prescripted gain map을 이용하여 균일하지 못한 음성 신호의 gain을 조절하는 알고리즘

3차년도 AGC 모델: STFT - GRU - MLP 의 경량화된 구조를 이용해 end-to-end로 음성 신호의 gain을 조절하는 알고리즘

Requirements
-------------
python==3.7.16     
torch==1.13.1+cu117                 
soundifle==0.12.1               
numpy==1.21.6       

Process
-------------
If you want to run version 2.1: Run AGC_endtoend/eval.ipynb

Training
-------------
Version 2.1: Run AGC_endtoend/train.py