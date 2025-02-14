import base64
import cv2
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

class VideoDescriptor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def description_video(self, video_path: str):
        VCap = cv2.VideoCapture(video_path)
        if not VCap.isOpened():
            raise FileNotFoundError("Файл не найден или имеет неподдерживаемый формат")
        
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

    def prompt_to_gpt(self, list_of_frame_base64):
        response = self.client.chat.completions.create(
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
            stream=True
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                time.sleep(0.1)


def main():
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("API ключ не найден в переменных окружения")
    
    print('Введите путь до видеофайла:')
    video_path = input().strip()
    
    descriptor = VideoDescriptor(api_key=API_KEY)
    frames_base64 = descriptor.description_video(video_path)
    descriptor.prompt_to_gpt(frames_base64)

if __name__ == "__main__":
    main()