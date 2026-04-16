from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.engine import TranslationEngine
from src.audio_utils import VADAudioBuffer
from fastapi.responses import HTMLResponse
import numpy as np

app = FastAPI()
engine = TranslationEngine()
vad_buffer = VADAudioBuffer(sample_rate=16000)

@app.get("/")
async def get_web_interface():
    # Nhớ đảm bảo file test_client.html nằm cùng thư mục gốc với thư mục src
    with open("test_client.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
@app.websocket("/meeting")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Meeting connected! Sẵn sàng nhận luồng âm thanh...")
    
    try:
        while True:
            # Nhận dữ liệu âm thanh
            data = await websocket.receive_bytes()
            audio_float = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Xử lý qua VAD
            completed_audio = vad_buffer.process_chunk(data, audio_float)
            
            if completed_audio:
                final_audio = np.frombuffer(completed_audio, dtype=np.int16).astype(np.float32) / 32768.0
                en_text, vi_text = await engine.process_audio(final_audio)
                
                if vi_text:
                    # Trả kết quả về cho giao diện
                    await websocket.send_json({
                        "original": en_text,
                        "translated": vi_text
                    })
                    print(f"EN: {en_text} -> VI: {vi_text}")

    except WebSocketDisconnect:
        print("Client đã chủ động ngắt kết nối.")
    except Exception as e:
        print(f"Lỗi WebSocket: {e}")
    finally:
        # Quan trọng: Không gọi websocket.close() ở đây nữa
        # Chỉ dọn dẹp bộ nhớ đệm
        vad_buffer.reset_buffer()