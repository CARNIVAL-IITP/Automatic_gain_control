# CARNIVAL_Automatic_Gain_Control
Carnival system을 구성하는 Automatic Gain Control 모델입니다.<br>
과학기술통신부 재원으로 정보통신기획평가원(IITP) 지원을 받아 수행한<br>
"원격 다자간 영상회의에서의 음성 품질 고도화 기술개발" 과제 공개 코드입니다.<br>
(2021.05~2024.12)<br>

본 Automatic gain control module은 Neural VAD와 prescripted gain map을 이용하여 균일하지 못한 음성 신호의 gain을 조절하는 알고리즘입니다.


Requirements
-------------
python==3.7.10     
torch==1.6.0                 
soundifle==0.10.3               
numpy==1.19.4       

Process
-------------
Run AGC.py

Training VAD
-------------
if you want to train Neural VAD module, check src/VAD.

