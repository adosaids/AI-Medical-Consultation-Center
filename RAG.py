import os
import requests
import io
from typing import List
from camel.storages import QdrantStorage
from ernie_adapters import ERNIEEmbedding
from camel.retrievers import VectorRetriever

# 设置千帆API密钥
ernie_api_key = os.environ.get("ERNIE_API_KEY", "")
ernie_secret_key = os.environ.get("ERNIE_SECRET_KEY", "")
os.environ["QIANFAN_ACCESS_KEY"] = ernie_api_key
os.environ["QIANFAN_SECRET_KEY"] = ernie_secret_key

embedding_instance = ERNIEEmbedding()

storage_instance = QdrantStorage(
    vector_dim=embedding_instance.get_output_dim(),
    path="E:/yibao/local_data",
    collection_name="camel_paper",
)
vector_retriever = VectorRetriever(embedding_model=embedding_instance,
                                   storage=storage_instance)
def RAG(question):
    retrieved_info = vector_retriever.query(
    query=question,
    top_k=1
    )
    return retrieved_info

def jian_1000(s):
    return len(s)>999

def cunchu_rag(msgs):
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
        "filename": filename,
        "total_chunks": len(chunks),
        "stored_chunks": stored_count,
        "text_length": len(text)
    }
# 改包了，把query的model.embed()改为model.do