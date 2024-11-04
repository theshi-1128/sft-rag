from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import json
import os


web_path_directory = "../dataset/web_paths_split_full.json"

# 从 JSON 文件中读取 web_paths
with open(web_path_directory, 'r') as file:
    data = json.load(file)
    web_paths = data["web_paths"]


model_name = "../embedding_model/bge-large-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="为这个句子生成表示以用于检索相关文章:"
            )

print("成功加载模型")
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(web_paths=tuple(web_paths))
docs = loader.load()
print(docs)


chunk_size = 1024
chunk_overlap = 512
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

vectorstore_directory = f"../vectorstore_{chunk_size}_{chunk_overlap}_split_full_bge_embedding"  # 选择一个目录来保存
# 确保目录存在
if not os.path.exists(vectorstore_directory):
    os.makedirs(vectorstore_directory)


splits = text_splitter.split_documents(docs)


db = FAISS.from_documents(splits, embedding=embedding)
db.save_local(vectorstore_directory)

print("向量数据库已成功保存。")
