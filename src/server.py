from fastapi import FastAPI, WebSocket
import uvicorn
import json

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected!")
    try:
        while True:
            # Nhận dữ liệu âm thanh thô từ trình duyệt
            data = await websocket.receive_bytes()
            
            # GIẢ LẬP: Ở đây ông sẽ gọi API của Deepgram/Groq
            # Hiện tại tôi trả về text giả lập để ông test kết nối
            await websocket.send_text(json.dumps({
                "type": "transcript",
                "text": "...đang nhận dữ liệu âm thanh..."
            }))
            await websocket.send_text(json.dumps({
                "type": "translation",
                "text": "...đang dịch..."
            }))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)