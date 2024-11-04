from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import json
import os
from langchain_core.documents import Document

web_path_directory = "../dataset/web_paths_split.json"
summary_file_path = "../dataset/summary.txt"  # 摘要文件的路径


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
# print(docs)





# 从文本文件中读取摘要
with open(summary_file_path, 'r', encoding='utf-8') as f:
    summaries = f.readlines()
    print(summaries)


# 将摘要与完整文档内容关联
summarized_docs = []
for doc, summary in zip(docs, summaries):
    # 去除摘要中的换行符并构建新的 Document
    summarized_docs.append(
        Document(
            metadata={
                'source': doc.metadata['source'],  # 保持源信息
                'summary': summary.strip()  # 将摘要存储在元数据中
            },
            page_content=doc.page_content
        )
    )
print(summarized_docs)




chunk_size = 1024
chunk_overlap = 512
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

vectorstore_directory = f"../vectorstore_{chunk_size}_{chunk_overlap}_with_summary_split_bge_embedding"  # 选择一个目录来保存
# 确保目录存在
if not os.path.exists(vectorstore_directory):
    os.makedirs(vectorstore_directory)


# 接下来使用 summarized_docs 进行文本分块和向量化
splits = text_splitter.split_documents(summarized_docs)

db = FAISS.from_documents(splits, embedding=embedding)
db.save_local(vectorstore_directory)

print("向量数据库已成功保存。")
