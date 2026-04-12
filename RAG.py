import os
import requests
import io
from typing import List
from camel.retrievers import VectorRetriever
import sys

# 设置千帆API密钥
ernie_api_key = os.environ.get("ERNIE_API_KEY", "")
ernie_secret_key = os.environ.get("ERNIE_SECRET_KEY", "")
os.environ["QIANFAN_ACCESS_KEY"] = ernie_api_key
os.environ["QIANFAN_SECRET_KEY"] = ernie_secret_key

# 全局变量
storage_instance = None
vector_retriever = None
embedding_instance = None
_qdrant_available = False

def init_storage():
    """延迟初始化存储，避免导入时出错"""
    global storage_instance, vector_retriever, embedding_instance, _qdrant_available

    if storage_instance is not None:
        return True

    try:
        from ernie_adapters import ERNIEEmbedding
        from camel.storages import QdrantStorage

        embedding_instance = ERNIEEmbedding()

        # 使用绝对路径并标准化路径分隔符
        local_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "local_data"))
        os.makedirs(local_data_path, exist_ok=True)

        # 使用内存模式避免文件锁问题
        storage_instance = QdrantStorage(
            vector_dim=embedding_instance.get_output_dim(),
            path=local_data_path,
            collection_name="camel_paper",
        )
        vector_retriever = VectorRetriever(
            embedding_model=embedding_instance,
            storage=storage_instance
        )
        _qdrant_available = True
        print(f"向量数据库初始化成功: {local_data_path}")
        return True
    except Exception as e:
        print(f"QdrantStorage 初始化失败: {e}")
        print("警告：RAG 功能将不可用")
        storage_instance = None
        vector_retriever = None
        embedding_instance = None
        _qdrant_available = False
        return False

def RAG(question):
    if not init_storage():
        return {"error": "向量检索器未初始化", "results": []}
    try:
        retrieved_info = vector_retriever.query(
            query=question,
            top_k=1
        )
        return retrieved_info
    except Exception as e:
        print(f"RAG 查询失败: {e}")
        return {"error": str(e), "results": []}

def jian_1000(s):
    return len(s) > 999

def cunchu_rag(msgs):
    if not init_storage():
        print("向量检索器未初始化，无法存储")
        return
    for msg in msgs:
        try:
            while jian_1000(msg):
                ms = msg[:999]
                vector_retriever.process(
                    content=ms,
                )
                msg = msg[999:]
            vector_retriever.process(
                content=msg,
            )
        except KeyboardInterrupt as e:
            print(f"出错{e}")
        except Exception as e:
            print(f"存储消息失败: {e}")

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """从PDF字节中提取文本"""
    try:
        import PyPDF2
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"PDF提取失败: {e}")
        raise Exception(f"PDF解析失败: {str(e)}")

def split_text_into_chunks(text: str, chunk_size: int = 999, overlap: int = 100) -> List[str]:
    """将文本分块，带重叠"""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def process_and_store_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """处理PDF并存储到向量数据库"""
    if not init_storage():
        return {
            "success": False,
            "filename": filename,
            "error": "向量检索器未初始化"
        }

    try:
        # 1. 提取文本
        text = extract_text_from_pdf_bytes(pdf_bytes)

        if not text.strip():
            raise Exception("PDF中没有提取到文本内容")

        # 2. 分块
        chunks = split_text_into_chunks(text)

        # 3. 存储到向量数据库
        stored_count = 0
        for i, chunk in enumerate(chunks):
            try:
                if chunk.strip():  # 跳过空块
                    vector_retriever.process(content=chunk)
                    stored_count += 1
            except Exception as e:
                print(f"存储第{i}块失败: {e}")
                continue

        return {
            "success": True,
            "filename": filename,
            "total_chunks": len(chunks),
            "stored_chunks": stored_count,
            "text_length": len(text)
        }
    except Exception as e:
        print(f"PDF处理失败: {e}")
        return {
            "filename": filename,
            "total_chunks": 0,
            "stored_chunks": 0,
            "text_length": 0,
            "error": str(e)
        }
