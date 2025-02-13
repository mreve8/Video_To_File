import base64
import cv2
import os
from dotenv import load_dotenv
from openai import OpenAI
import time

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

def description_video(VCap):
    list_of_frame_base64 = []

    while VCap.isOpened():
        ret, frame = VCap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.jpg', frame)
        list_of_frame_base64.append(base64.b64encode(buffer).decode("utf-8"))

    VCap.release()
    
    if not list_of_frame_base64:
        raise ValueError("Не удалось преобразовать кадры в видео")

    return list_of_frame_base64

def prompt_to_gpt(list_of_frame_base64):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Создай подробное описание происходящего в видео"},
                *[
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
                    for frame in list_of_frame_base64[0::50]
                ],
            ],
        }
        ],
        temperature=0.6,
        stream = True
        )
    
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            time.sleep(0.1)

try:
    print('Введите путь до ведеофайла:')
    video_path = str(input())
    VCap = cv2.VideoCapture(f'{video_path}')

    if not VCap.isOpened():
        raise FileNotFoundError("Файл не найден или имеет неподдерживаемый формат")
    
    prompt_to_gpt(description_video(VCap))

except FileNotFoundError as e:
    print(f"Ошибка: {e}")