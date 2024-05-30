import easyocr  # 텍스트 인식을 위한 모듈
import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from googletrans import Translator  # 텍스트 번역을 위한 모듈

app = Flask(__name__)

# EasyOCR 리더 초기화 - 인식할 언어 지정
reader = easyocr.Reader(["en", "ko"], gpu=True)  # 영어, 한국어
reader_ja = easyocr.Reader(["ja"], gpu=True)  # 일본어
reader_ch = easyocr.Reader(["ch_sim"], gpu=True)  # 중국어

# 웹캠
cap = cv2.VideoCapture(0)  # 0은 내장 웹캠을 사용하는 경우입니다.

# Google Translate 초기화
translator = Translator()


# 카메라 화면 구현, 인식되는 텍스트에 직사각형 그리기
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 텍스트 인식
        result = reader.readtext(frame)
        result_ja = reader_ja.readtext(frame)
        result_ch = reader_ch.readtext(frame)

        THRESHOLD = 0.5

        for bbox, text, conf in result + result_ja + result_ch:
            if conf > THRESHOLD:
                # 경계 상자 좌표 추출
                pt1 = tuple(map(int, bbox[0]))
                pt2 = tuple(map(int, bbox[2]))
                # 직사각형 그리기
                cv2.rectangle(frame, pt1, pt2, color=(0, 0, 255), thickness=3)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# 인식된 텍스트 반환
@app.route("/get_recognized_text")
def get_recognized_text():
    ret, frame = cap.read()
    if not ret:
        return "카메라에서 이미지를 가져올 수 없습니다."

    result = reader.readtext(frame)
    result_ja = reader_ja.readtext(frame)
    result_ch = reader_ch.readtext(frame)

    THRESHOLD = 0.3
    recognized_text = []

    for bbox, text, conf in result + result_ja + result_ch:
        if conf > THRESHOLD:
            recognized_text.append(text)

            pt1 = tuple(map(int, bbox[0]))
            pt2 = tuple(map(int, bbox[2]))
            cv2.rectangle(frame, pt1, pt2, color=(0, 0, 255), thickness=3)

    return " ".join(recognized_text)


# 인식한 텍스트 번역하기
@app.route("/translate", methods=["POST"])
def translate_text():
    # 카메라로 인식한 텍스트 가져오기
    ret, frame = cap.read()
    if not ret:
        return "카메라에서 이미지를 가져올 수 없습니다."

    result = reader.readtext(frame)
    result_ja = reader_ja.readtext(frame)
    result_ch = reader_ch.readtext(frame)

    THRESHOLD = 0.3
    recognized_text = []

    for bbox, text, conf in result + result_ja + result_ch:
        if conf > THRESHOLD:
            recognized_text.append(text)

    # 인식한 텍스트를 하나의 문자열로 결합
    full_recognized_text = " ".join(recognized_text)

    # 클라이언트에서 선택한 언어 가져오기
    select_language = request.json.get("select_language")
    print("Selected Language: ", select_language)

    # 선택한 언어 코드로 변환
    language_codes = {"한국어": "ko", "영어": "en", "중국어": "zh-cn", "일본어": "ja"}
    target_language_code = language_codes.get(select_language)

    if target_language_code:  # 언어 코드가 존재하면 번역하여 텍스트를 반환
        translated_text = translator.translate(
            full_recognized_text, dest=target_language_code
        ).text  # 매개변수: 번역할 텍스트, 언어 코드
        print("번역된 텍스트: ", translated_text)
        return translated_text
    else:
        return recognized_text  # 선택된 언어가 없으면 원본 텍스트를 반환


if __name__ == "__main__":
    app.run(debug=True)
