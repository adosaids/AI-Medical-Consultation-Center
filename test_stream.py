import os
import asyncio
from ernie_types import ModelType
from factor import ModelFactory

async def test_stream():
    model = ModelFactory.create(
        model_type=ModelType.DEEPSEEK_V3_1_250821,
        model_config_dict={"temperature": 0.5},
        api_key=os.environ.get("ERNIE_API_KEY", ""),
        sk=os.environ.get("ERNIE_SECRET_KEY", ""),
        liu=True
    )
    
    messages = [{"role": "user", "content": "你好"}]
    prompt = "你是医疗助手"
    
    print("开始流式调用...")
    stream = model.run(messages, prompt=prompt)
    
    for i, chunk in enumerate(stream):
        print(f"Chunk {i}: {repr(chunk)}")
        if i > 5:  # 只打印前6个
            break

if __name__ == "__main__":
    asyncio.run(test_stream())
