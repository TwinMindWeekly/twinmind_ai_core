import os
from dotenv import load_dotenv
from twinmind_ai_core.factory import AIProviderFactory

# Đảm bảo bạn đã cấu hình .env (ví dụ: copy .env từ thư mục comic_translator sang đây để test)
load_dotenv()

def test_stream():
    print("Initializing Multi-LLM test...")
    # Khởi tạo qua AutoFallback
    provider = AIProviderFactory.get_provider("auto")
    
    print(f"Using Provider: {provider.get_usage_info()}")
    print("-" * 50)
    
    system_prompt = "You are a friendly AI."
    user_prompt = "Tell me a short joke."
    
    print("Testing Stream...")
    stream = provider.generate_text(system_prompt=system_prompt, user_prompt=user_prompt, stream=True)
    
    if isinstance(stream, str):
        print(stream)
    else:
        for chunk in stream:
            print(chunk, end="", flush=True)
            
    print("\n")
    print("-" * 50)
    print(f"Post usage info: {provider.get_usage_info()}")

if __name__ == "__main__":
    test_stream()
