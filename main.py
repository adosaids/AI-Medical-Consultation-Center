from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from camel.messages import BaseMessage
from camel.types import RoleType
from ernie_types import ModelType
from agent import ErnieChineseAgent
import uvicorn
from fastapi.staticfiles import StaticFiles
import os

from RAG import process_and_store_pdf
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
from work import work
system_message = BaseMessage(
    role_name="system",
    role_type=RoleType.ASSISTANT,
    content="你是一个智能助手，负责回答用户的问题。",
    meta_dict=[{"a":"b"}]
    )
prompt = []
    # 从环境变量获取 ERNIE API Key 和 Secret Key
ernie_api_key = os.environ.get("ERNIE_API_KEY", "")
ernie_secret_key = os.environ.get("ERNIE_SECRET_KEY", "")

    # 初始化 ErnieChineseAgent
agent = ErnieChineseAgent(
        system_message=system_message,
        ernie_api_key=ernie_api_key,
        ernie_secret_key=ernie_secret_key,
        model_type = ModelType.ERNIE_8K,
    )
zhenduantuili_type = [ModelType.ERNIE_8K,ModelType.ERNIE_8K]
zhiliaoguihua_type = [ModelType.ERNIE_8K,ModelType.ERNIE_8K]
work_liu = work(agent_jiekou=agent,zhenduantype=zhenduantuili_type,zhiliaotype=zhiliaoguihua_type)
@app.post("/huanzhejiekou/{wenti}")
async def huanzhejiekou(wenti: str):
    mes = BaseMessage(
        role_name="患者",
        role_type=RoleType.USER,
        content=wenti,
        meta_dict={"患者接口":"输入"}
    )
    res = agent.step(mes,rolename='患者接口',prompt="".join(prompt))
    return {
        "护士": res
    }

@app.post("/zhenduantuili/{request}")
async def zhenduantuili(request: str):
    li_tuili = work_liu.work()
    li_guihua = work_liu.work_guihua(request)
    return {
        "诊断推理":li_tuili,
        "治疗规划":li_guihua
    }



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1",port=8000,reload=True)
# 前端联调流式输出
# rag模块的接口设计
# rag作为一个工具给agent调用，agent调用rag接口，rag接口返回检索到的信息，agent根据检索到的信息进行回答
# 长期记忆要改，目前是短期记忆，每次来都是新用户，这里要改成redis来保存用户之前的基本信息和症状信息
# 用户症状要改，不能仅通过提示词拼接，要专门写一个类来保存用户的症状信息，agent能看
# 治疗规划要改，能够展示每一板治疗规划，重新处理逻辑，要先生成一板，问用户有没有更多诉求