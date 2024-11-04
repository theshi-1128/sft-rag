import os
import json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ["OPENAI_API_KEY"] = "sk-eUazPxq20iyLu9W0A63eE1Ff71Eb4b0885D9D88cF3Ff2204"
os.environ["OPENAI_API_BASE"] = "https://4.0.wokaai.com/v1"
vectorstore_directory = "../vectorstore_1024_512_openai_embedding"  # 选择一个目录来保存
# 确保目录存在
if not os.path.exists(vectorstore_directory):
    os.makedirs(vectorstore_directory)


web_path_directory = "../dataset/web_paths.json"

# 从 JSON 文件中读取 web_paths
with open(web_path_directory, 'r') as file:
    data = json.load(file)
    web_paths = data["web_paths"]

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(web_paths=tuple(web_paths))
docs = loader.load()
# print(docs)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=512)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=vectorstore_directory)