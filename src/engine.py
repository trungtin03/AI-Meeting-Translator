import asyncio
from faster_whisper import WhisperModel
from transformers import AutoTokenizer
import ctranslate2
from huggingface_hub import snapshot_download

class TranslationEngine:
    def __init__(self):
        # Tự động nhận diện GPU hoặc chạy CPU nếu không có
        self.device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        self.compute_type = "int8" 
        
        print(f"--- Đang khởi tạo AI Engine trên thiết bị: {self.device.upper()} ---")

        # 1. Não Nghe (STT): Vẫn giữ nguyên Whisper nhưng đã được buff độ thông minh
        self.stt_model = WhisperModel("distil-small.en", device=self.device, compute_type=self.compute_type)
        
        # 2. Não Dịch: Đổi sang CHUYÊN GIA ANH - VIỆT (Helsinki-NLP)
        print("Đang tải model dịch chuyên sâu Anh-Việt (Helsinki-NLP)...")
        
        # Gọi thẳng model từ thư mục mày vừa tự ép xung trên máy
        self.translator = ctranslate2.Translator("opus-mt-en-vi-ct2", device=self.device, compute_type=self.compute_type)
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
        
        print("--- Tất cả Models đã sẵn sàng! ---")

    async def process_audio(self, audio_chunk):
        # Mớm trước ngữ cảnh hoặc từ chuyên ngành nếu cần (Ví dụ: "AI, machine learning")
        my_glossary = "" 
        
        segments, _ = self.stt_model.transcribe(
            audio_chunk, 
            language="en", 
            beam_size=3, 
            condition_on_previous_text=False,
            initial_prompt=my_glossary
        )
        
        english_text = "".join([segment.text for segment in segments]).strip()
        if not english_text: return None, None
        
        # Đẩy sang cho thằng Helsinki dịch
        vietnamese_text = self.translate_en_to_vi(english_text)
        return english_text, vietnamese_text

    def translate_en_to_vi(self, text):
        # Đóng gói câu tiếng Anh
        source = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        
        # Dịch thẳng (Helsinki chỉ biết 1 đường Anh->Việt nên không cần chỉ định mã ngôn ngữ như NLLB)
        results = self.translator.translate_batch([source])
        
        # Mở gói kết quả
        target_tokens = results[0].hypotheses[0]
        
        # Ghép từ lại thành câu hoàn chỉnh và tự động xóa bỏ các thẻ hệ thống rác
        final_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target_tokens), skip_special_tokens=True)
        return final_text