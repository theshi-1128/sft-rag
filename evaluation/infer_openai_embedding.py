# from data_generating.getting_response_with_rag_bge_embedding import process_jsonl
from data_generating.getting_response_with_mixed_rag_bge_embedding import process_jsonl


if __name__ == '__main__':
    model = "zhipu"  # openai或zhipu或qwen
    input_jsonl_file = '../dataset/test1.jsonl'  # 输入文件名
    output_jsonl_file = f'../dataset/answer_zh3_1024_512_5_bge-reranker-v2-m3_glm-4-flash:499254306::auoms6xs_mixed_retriever.jsonl'  # 输出文件名
    process_jsonl(input_jsonl_file, output_jsonl_file, model)
