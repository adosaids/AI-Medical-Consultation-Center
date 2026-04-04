import json
import PyPDF2
import os
import glob
from tqdm import tqdm
def wenjian(path):
    try:
        with open(path,'r',encoding='utf-8') as f:
           if '.json' in path:
               data =  json.load(f)
               li = dict(data)   
    except Exception as e:
        print(f"处理文件{path}时出错:{e}")
        return None
    return li

def get_model_encoding(value_for_tiktoken: str):
    r"""Get model encoding from tiktoken.

    Args:
        value_for_tiktoken: Model value for tiktoken.

    Returns:
        tiktoken.Encoding: Model encoding.
    """
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(value_for_tiktoken)
    except KeyError:
        print("Model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return encoding

def prompts(paths = ["E:/yibao/prompt/Role.json","E:/yibao/prompt/phase.json","E:/yibao/prompt/chatchain.json"]):
    prompts_role = wenjian(paths[0])
    prompts_phase = wenjian(paths[1])
    chatchain = wenjian(paths[2])
    return prompts_role,prompts_phase,chatchain

def RAG_xinxi_cunru(pdf_path: str = 'E:/bot/第十版人卫版本本科教材'):
    res = []
    for filepath in tqdm(glob.glob(os.path.join(pdf_path, '*.pdf'))):
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            res.append(text)
        except Exception as e:
            print(f"PyPDF2 提取失败: {e}")
            return res
    return res