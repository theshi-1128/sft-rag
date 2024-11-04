from llm.api import get_response, get_zhipu_response, get_gpt_response
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagReranker
from tqdm import tqdm
import json


model_name = "../embedding_model/bge-large-zh-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="为这个句子生成表示以用于检索相关文章:"
            )

reranker = FlagReranker('../embedding_model/bge-reranker-v2-m3', use_fp16=True)  # Setting use_fp16 to True speeds up



# 从本地加载 FAISS 索引
db = FAISS.load_local(
    '../vectorstore_1024_512_bge_embedding',
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 3})


def generate_answer(context, question, answer, model_type="openai"):
    optimize_rag_answer_prompt = (
        "任务：请判断当前对用户问题的回答是否符合 TuGraph-DB 知识库中的检索到的相关内容描述。如果不符合，请基于检索到的内容对当前回答进行进一步优化。\n"
        "最多只用三句话回答，确保优化后的回答简明扼要。"
        "\n\n"
        f"用户问题：{question}\n"
        f"当前回答：{answer}\n"
        f"检索到的相关内容: {context}\n"
        "优化后的回答:"
    )
    if model_type == "openai":
        return get_gpt_response(optimize_rag_answer_prompt)
    elif model_type == "zhipu":
        return get_zhipu_response(optimize_rag_answer_prompt)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_rag(question, answer, model="zhipu"):
    # 检索相关文档
    docs = retriever.invoke(answer)
    contexts_list = [doc.page_content for doc in docs]

    query_passages = [[answer, passage] for passage in contexts_list]
    scores = reranker.compute_score(query_passages, normalize=True)
    # 根据得分对段落进行排序（从高到低）
    sorted_passages = sorted(zip(scores, contexts_list), key=lambda x: x[0], reverse=True)

    # 提取rerank后的段落
    sorted_contexts = "\n".join([passage for score, passage in sorted_passages])

    # 生成回答
    opt_answer = generate_answer(sorted_contexts, question, answer, model)

    return sorted_contexts, opt_answer


def optimize_questions(input_question_file, input_answer_file, output_file, model="openai"):
    with open(input_question_file, 'r', encoding='utf-8') as qfile, \
         open(input_answer_file, 'r', encoding='utf-8') as afile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        # 使用 zip 同时迭代两个文件
        for qline, aline in tqdm(zip(qfile, afile)):
            qdata = json.loads(qline.strip())  # 去除末尾换行符并解析 JSON
            adata = json.loads(aline.strip())  # 去除末尾换行符并解析 JSON
            question = qdata['input_field']
            answer = adata['output_field']
            # 获取答案
            context, opt_answer = run_rag(question, answer, model)
            print("opt_answer:", opt_answer)
            # 创建新的字典以保存优化后的问题
            optimized_data = {
                'id': adata['id'],
                'output_field': opt_answer
            }
            # 写入到输出文件
            outfile.write(json.dumps(optimized_data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # # 示例问题
    # question = "创建新子图时需要哪些参数？"
    # answer = "创建新子图时需要填写子图名称、子图描述和配置信息。"
    # # 获取答案
    # context, opt_answer = run_rag(question, answer, "zhipu")  # openai或zhipu
    # print("Context:", context)
    # print("Opt_answer:", opt_answer)
    input_question_path = '../dataset/test1.jsonl'
    input_answer_path = '../dataset/answer.jsonl'
    output_file = '../dataset/optimized_answer.jsonl'
    optimize_questions(input_question_path, input_answer_path, output_file, model="zhipu")
