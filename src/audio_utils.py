import torch
import numpy as np

class VADAudioBuffer:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.buffer = bytearray()
        self.is_speaking = False
        self.silence_frames = 0
        # Mới: Giới hạn buffer tối đa (ví dụ 5-7 giây) để tránh "ngâm" quá lâu
        self.max_buffer_size = sample_rate * 2 * 7 # 7 giây (2 bytes mỗi sample)
        
        print("Đang tải Silero VAD...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        print("VAD Ready!")

    def is_speech(self, audio_data: np.ndarray) -> bool:
        audio_tensor = torch.from_numpy(audio_data)
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        # Giảm ngưỡng xuống 0.4 để nhạy hơn với tiếng video
        return speech_prob > 0.4 

    def process_chunk(self, chunk: bytes, chunk_float: np.ndarray):
        has_speech = self.is_speech(chunk_float)

        if has_speech:
            self.buffer.extend(chunk)
            self.is_speaking = True
            self.silence_frames = 0
            
            # MỚI: Nếu nói quá lâu (quá 7s) -> Ép xả đi dịch luôn
            if len(self.buffer) > self.max_buffer_size:
                completed_audio = self.buffer.copy()
                self.reset_buffer()
                return completed_audio
        else:
            if self.is_speaking:
                self.silence_frames += 1
                self.buffer.extend(chunk)
                
                # Nếu im lặng quá 2 frames (nhanh hơn bản cũ) -> Dịch ngay
                if self.silence_frames > 2:
                    completed_audio = self.buffer.copy()
                    self.reset_buffer()
                    return completed_audio

        return None

    def reset_buffer(self):
        self.buffer = bytearray()
        self.is_speaking = False
        self.silence_frames = 0