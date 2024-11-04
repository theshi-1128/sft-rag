from typing import Dict, List, Tuple


def reciprocal_rank_fusion(merged_contexts: List[str], merged_scores: List[float], k: int = 5) -> Tuple[List[str], List[float]]:
    """
    Perform Reciprocal Rank Fusion (RRF) on the provided search results.

    Args:
        merged_contexts (List[str]): A list of document contents.
        merged_scores (List[float]): A list of corresponding scores for the documents.
        k (int): A constant added to the rank for score adjustment (default is 60).

    Returns:
        Tuple[Dict[str, float], List[str], List[float]]: A tuple containing:
            - A dictionary of document contents with their fused scores, sorted by score.
            - A list of documents (sorted).
            - A list of corresponding fused scores (sorted).
    """
    fused_scores = {}

    # 创建一个字典，将上下文和得分配对
    contexts_with_scores = {context: score for context, score in zip(merged_contexts, merged_scores)}

    for rank, (doc, score) in enumerate(sorted(contexts_with_scores.items(), key=lambda x: x[1], reverse=True)):
        if doc not in fused_scores:
            fused_scores[doc] = 0
        # 更新融合得分
        fused_scores[doc] += 1 / (rank + k)

    # 按照融合得分降序排序
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # 提取排序后的文档和得分
    sorted_contexts = [doc for doc, score in sorted_results]
    sorted_scores = [score for doc, score in sorted_results]

    return sorted_contexts, sorted_scores





if __name__ == '__main__':
    # 这个方法接受一个字典，其中每个键是查询字符串，值是一个包含文档得分的字典。你可以调用这个方法，如下所示：

    final_sorted_contexts = ["你啊后", "哈哈哈", "cnm", "dsafwfa"]

    final_sorted_scores = [0.42, 0.11, 0.93, 0.78]

    sorted_contexts, sorted_scores = reciprocal_rank_fusion(final_sorted_contexts, final_sorted_scores, k=5)
    print(sorted_contexts)
    print(sorted_scores)
    # 这个方法可以灵活地处理不同的查询和文档得分，返回融合后的文档得分结果。