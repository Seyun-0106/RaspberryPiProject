from __future__ import division

import re
import sys
import os
from tkinter import *
import tkinter
import tkinter.font
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import webbrowser
import time

import pyaudio
from six.moves import queue
import speech_recognition as sr

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from hangul_utils import join_jamos

user_name = "캡스톤"
user_birth = "1999년 1월 1일"
user_work = "학생"
user_num = "010-1234-5678"

gesture = {
    26:'spacing', 27:'clear',
    28:'ㄱ', 29:'ㄴ',30:'ㄷ',31:'ㄹ',32:'ㅁ',33:'ㅂ',34:'ㅅ',35:'ㅇ',36:'ㅈ',37:'ㅊ',38:'ㅋ',39:'ㅌ',40:'ㅍ',41:'ㅎ',
    42:'ㅏ',43:'ㅑ',44:'ㅓ',45:'ㅕ',46:'ㅗ',47:'ㅛ',48:'ㅜ', 49:'ㅠ',50:'ㅡ',51:'ㅣ',52:'ㅐ',53:'ㅒ',54:'ㅔ',55:'ㅖ',
    56:'ㅢ',57:'ㅚ',58:'ㅟ',59:'반가워요',60:'만나서',61:'네', 62:'아니오', 63:'감사합니다'
}
def sign_language():
    cap = cv2.VideoCapture(0)
    cap.set(3,320)
    cap.set(4,240)
    max_num_hands = 2
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

    file = np.genfromtxt('/home/pi/Desktop/dataset.txt', delimiter=',')
    angleFile = file[:,:-1] #좌표
    labelFile = file[:,-1]  #label
    angle = angleFile.astype(np.float32)
    label = labelFile.astype(np.float32)
    knn = cv2.ml.KNearest_create() #모델 훈련시켜
    knn.train(angle,cv2.ml.ROW_SAMPLE,label)
    

    startTime = time.time()
    prev_index = 0
    sentence = ''
    recognizeDelay = 2

    while True:
        ret, img = cap.read()
        img = cv2.flip(img,1) #좌우반전

    
        if not ret:
            continue
        imgRGB = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        result = hands.process(imgRGB)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21,3))
                for j,lm in enumerate(res.landmark):
                    joint[j] = [lm.x,lm.y,lm.z]

            # 각도 계산        
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],:]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15, 16, 17, 18, 19, 20],:]

            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:,np.newaxis]
            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13 ,14, 16, 17], :]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],:]
            angle = np.arccos(np.einsum('nt,nt->n',compareV1,compareV2))
            angle = np.degrees(angle)

            data = np.array([angle],dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data,3)
            index = int(results[0][0])
            if index in gesture.keys():
                if index!=prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay:
                        if index == 26: #space
                            sentence += ' '
                        elif index == 27:   #clear
                            #sentence = ''
                            cv_str = join_jamos(sentence)
                            btn_cv.config(text=cv_str)
                            get_text4()
                            break
                        else:
                            sentence += gesture[index]
                        startTime = time.time()
                        
                font_cv = ImageFont.truetype("/home/pi/Desktop/SCDream6.otf", 20)
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)

                draw.text((2,70),gesture[index],font=font_cv, color = (255,255,255))
                
                img = np.array(img_pil)
                
            mp_drawing.draw_landmarks(img,res,mp_hands.HAND_CONNECTIONS)

        # 화면에 텍스트 출력\
        cv2.rectangle(img=img, pt1=(2, 40), pt2=(100, 57), color=(0,0,0), thickness=-1)
        font_cv = ImageFont.truetype("/home/pi/Desktop/SCDream6.otf", 20)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        draw.text((2,40),sentence,font=font_cv,color =(255,255,255))
        
        img = np.array(img_pil)

        cv2.imshow('HandTracking',img)
        cv2.waitKey(1)
        

def stt_recognize():    
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak Anything :")
        audio = r.listen(source, timeout=3, phrase_time_limit=3)
        try:
            text = r.recognize_google(audio,language='ko')
            sendText(text)
        except:
            print("Sorry could not recognize what you said")

#수정
def sendText(txt): 
    text_list = txt.split(" ")

    for i in range(len(text_list)):
        if text_list[i]=="영수증":
            i=i+1
            if text_list[i]=="드릴까요":
                btn_1.config(text="현금영수증 주세요")
                btn_2.config(text="네")
                btn_3.config(text="괜찮습니다")
            elif text_list[i]=="줄까요":
                btn_1.config(text="현금영수증 주세요")
                btn_2.config(text="네")
                btn_3.config(text="괜찮아요")
            elif text_list[i]=="필요하세요":
                btn_1.config(text="현금영수증 주세요")
                btn_2.config(text="네")
                btn_3.config(text="괜찮습니다")         
                
        if text_list[i]=="연락처": 
            i=i+1
            if text_list[i]=="드릴까요":
                btn_1.config(text="네")
                btn_2.config(text="아니요")
                btn_3.config(text="적어서 주세요")      
            if text_list[i]=="주세요":
                btn_1.config(text=user_num + "입니다")
                btn_2.config(text="적어서 드릴게요")
                btn_3.config(text="알려드리기 곤란합니다")   

        if text_list[i]=="번호": 
            i=i+1
            if text_list[i]=="드릴까요":
                btn_1.config(text="네")
                btn_2.config(text="아니요")
                btn_3.config(text="적어서 주세요")      
            if text_list[i]=="주세요":
                btn_1.config(text=user_num + "입니다")
                btn_2.config(text="적어서 드릴게요")
                btn_3.config(text="알려드리기 곤란합니다")        

        if text_list[i]=="드시고": 
            i=i+1
            if text_list[i]=="가세요":
                btn_1.config(text="네")
                btn_2.config(text="아니요")
                btn_3.config(text="포장 부탁드려요")

        if text_list[i]=="적립": 
            i=i+1
            if text_list[i]=="하시겠어요":
                btn_1.config(text="아니요")
                btn_2.config(text="번호 누를게요")
                btn_3.config(text="카드 드릴게요")      
            if text_list[i]=="하실까요":
                btn_1.config(text="아니요")
                btn_2.config(text="번호 누를게요")
                btn_3.config(text="카드 드릴게요")
            if text_list[i]=="하세요":
                btn_1.config(text="아니요")
                btn_2.config(text="번호 누를게요")
                btn_3.config(text="카드 드릴게요")
                     

        if text_list[i]=="이름이": 
            i=i+1
            if text_list[i]=="어떻게":
                i=i+1
                if text_list[i]=="되세요":
                    btn_1.config(text=user_name+" 입니다")
                    btn_2.config(text="적어드릴게요")   
                    btn_3.config(text="알려드리기 곤란합니다")
            if text_list[i]=="뭐예요":
                btn_1.config(text=user_name+" 입니다")
                btn_2.config(text="적어드릴게요") 
                btn_3.config(text="알려드리기 곤란합니다")
            if text_list[i]=="뭐야":
                btn_1.config(text=user_name+" 이야")
                btn_2.config(text="적어줄게") 

        if text_list[i]=="성함이": 
            i=i+1
            if text_list[i] == "어떻게":
                i=i+1
                if text_list[i] == "되세요":
                    btn_1.config(text=user_name+" 입니다")
                    btn_2.config(text="적어드릴게요")   
                    btn_3.config(text="알려드리기 곤란합니다")
            if text_list[i]=="뭐예요":
                btn_1.config(text=user_name+" 입니다")
                btn_2.config(text="적어드릴게요") 
                btn_3.config(text="알려드리기 곤란합니다")
                
        if text_list[i]=="생년월일이": 
            i=i+1
            if text_list[i] == "어떻게":
                i=i+1
                if text_list[i] == "되세요":
                    btn_1.config(text=user_birth+" 입니다")
                    btn_2.config(text="적어드릴게요")     
                    btn_3.config(text="알려드리기 곤란합니다.")
            if text_list[i]=="뭐예요":
                btn_1.config(text=user_birth+" 입니다")
                btn_2.config(text="적어드릴게요") 
                btn_3.config(text="알려드리기 곤란합니다.")

        if text_list[i]=="생일이": 
            i=i+1
            if text_list[i] == "어떻게":
                i=i+1
                if text_list[i] == "되세요":
                    btn_1.config(text=user_birth+" 입니다")
                    btn_2.config(text="적어드릴게요")     
                    btn_3.config(text="알려드리기 곤란합니다.")
            if text_list[i]=="뭐예요":
                btn_1.config(text=user_birth+" 입니다")
                btn_2.config(text="적어드릴게요") 
                btn_3.config(text="알려드리기 곤란합니다.")

        if text_list[i] == "무슨":
            i=i+1
            if text_list[i] == "일":
                i=i+1
                if text_list[i] == "하세요":
                    btn_1.config(text=user_work+" 입니다")
                    btn_2.config(text="적어드릴게요")      
                    btn_3.config(text="알려드리기 곤란합니다.")

    message_speech.config(text=txt)

def get_text1():
    txt = btn_1['text']
    tts1(txt)

def get_text2():
    txt = btn_2['text']
    tts2(txt)    

def get_text3():
    txt = btn_3['text']
    tts3(txt)

def get_text3():
    txt = btn_3['text']
    tts3(txt)
    
def get_text4():
    txt = btn_cv['text']
    tts4(txt)

def tts1(speak_text):
    tts = gTTS(text=speak_text, lang='ko')
    filename='speech1.mp3'
    tts.save(filename)
    s1 = AudioSegment.from_file(filename)
    play(s1)
    os.remove(filename)

def tts2(speak_text):
    tts = gTTS(text=speak_text, lang='ko')
    filename='speech2.mp3'
    tts.save(filename)
    s2 = AudioSegment.from_file(filename)
    play(s2)
    os.remove(filename)

def tts3(speak_text):
    tts = gTTS(text=speak_text, lang='ko')
    filename='speech3.mp3'
    tts.save(filename)
    s3 = AudioSegment.from_file(filename)
    play(s3)      
    os.remove(filename)
    
def tts4(speak_text):
    tts = gTTS(text=speak_text, lang='ko')
    filename='speech4.mp3'
    tts.save(filename)
    s4 = AudioSegment.from_file(filename)
    play(s4)      
    os.remove(filename)

def clear():
    btn_1.config(text="저는 청각장애인입니다")
    btn_2.config(text="도와주세요")
    btn_3.config(text="다시 한번 말씀해 주세요")

window=tkinter.Tk()
window.title("Psyren")
window.geometry("800x480+100+100")
window.resizable(False, False)

font = tkinter.font.Font(family="맑은 고딕", size=17)
font1 = tkinter.font.Font(family="맑은 고딕", size=17)
font2 = tkinter.font.Font(family="맑은 고딕", size=11)

def start_to_stt():
    Page_stt.pack()
    Page_start.forget()

def stt_to_start():
    Page_stt.forget()
    Page_start.pack()\

def open():
    webbrowser.open('https://quirky-swartz-146cfb.netlify.app/')

#수정    
def digital_clock():
    time_live = time.strftime("%H:%M:%S")
    label_time.config(text=time_live) 
    label_time.after(200, digital_clock)

def press():
    cv2.destroyAllWindows()

# start 프레임
Page_start = tkinter.Frame(window)
Page_start.configure( width=800, height=480)
Page_start.pack_propagate(0)
Page_start.pack()
Page_start.configure(bg='#FFFDCC')
label_time = tkinter.Label(Page_start, font = ("맑은 고딕", 50, "bold"), background="#FFFDCC")
label_time.place(x=260, y=150)
digital_clock()

btn_stt = tkinter.Button(Page_start, borderwidth = 1, font=font2, text="대화하기",background = "#B5D9B2", relief="ridge", width=20, height=2, command=start_to_stt)
btn_stt.place(x=160, y=330)
btn_audio = tkinter.Button(Page_start, borderwidth = 1, font=font2, text="소리듣기", background = "#B5D9B2", relief="ridge", width=20, height=2, command=open)
btn_audio.place(x=430, y=330)

# stt 프레임
Page_stt = tkinter.Frame(window)
Page_stt.configure(width=800, height=480)
Page_stt.pack_propagate(0)
Page_stt.configure(bg='#FFFDCC')

message_speech=tkinter.Message(Page_stt, text="", borderwidth = 0, font = font, background = "#FFFDCC", width=400, relief="ridge")
message_speech.place(x=60,y=60)
btn_pre = tkinter.Button(Page_stt, text="이전 페이지", borderwidth = 1, relief="ridge",background = "#C6DDC1",width=8, height=1, command=stt_to_start)
btn_pre.place(x=60,y=330)
btn_start_speech = tkinter.Button(Page_stt, command=stt_recognize, borderwidth = 1, text="음성인식", background = "#B5D9B2", relief="ridge", width=15, height=2)
btn_start_speech.place(x=170,y=320)

btn_1 = tkinter.Button(Page_stt, command=get_text1, borderwidth = 1, text="저는 청각장애인입니다", background = "#F2C6A0", relief="ridge", width=45, height=2)
btn_1.place(x=50,y=140)
btn_2 = tkinter.Button(Page_stt, command=get_text2, borderwidth = 1, text="도와주세요", background = "#F2C6A0", relief="ridge", width=45, height=2)
btn_2.place(x=50,y=190)
btn_3 = tkinter.Button(Page_stt, command=get_text3, borderwidth = 1, text="다시 한번 말씀해 주세요", background = "#F2C6A0", relief="ridge", width=45, height=2)
btn_3.place(x=50,y=240)

btn_start = tkinter.Button(Page_stt, font=font2, command=sign_language, borderwidth = 1, text="수어 번역", background = "#B5D9B2", relief="ridge", width=7, height=2)
btn_start.place(x=530,y=300)
btn_stop = tkinter.Button(Page_stt, font=font2, command=press, borderwidth = 1, text="종료", background = "#B5D9B2", relief="ridge", width=7, height=2)
btn_stop.place(x=630,y=300)

btn_cv = tkinter.Button(Page_stt, font=font2, command=get_text4, borderwidth = 1, text="", background = "#F2C6A0", relief="ridge", width=25, height=2)
btn_cv.place(x=500,y=200)

window.mainloop()


