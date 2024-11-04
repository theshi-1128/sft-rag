import os
from openai import OpenAI
from zhipuai import ZhipuAI

def get_gpt_response(prompt):
    # 设置 API 密钥
    os.environ["OPENAI_API_KEY"] = "sk-eUazPxq20iyLu9W0A63eE1Ff71Eb4b0885D9D88cF3Ff2204"
    os.environ["OPENAI_API_BASE"] = "https://4.0.wokaai.com/v1"

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "请帮我把下面的代码添加上中文解释。请不要对原始内容进行改动。输出保持原来的markdown标准格式。记住！保留原始的英文注释，在英文注释后面加上中文的解释。记住！不要让中文解释把英文注释覆盖了。请确保完整保留原始内容。"},
                {"role": "user", "content": prompt}
            ],
            model="gpt-4o-mini",
            # top_p=0,
            # temperature=0,
            max_tokens=8192,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return str(e)


def get_response(prompt):
    # 设置 API 密钥
    client = ZhipuAI(api_key="0423d9de045f1539087b88fc9367e0ba.ZJlRZY4RxsTrCME9")
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",  #glm-4-flash, glm-4-plus,
            # glm-4-9b:499254306::9rjsdakn (第一次的sft模型 1epoch bs8)
            # glm-4-flash:499254306::mr2ahrnw (第一次的sft模型 1epoch bs8)
            # glm-4-flash:499254306::auoms6xs (第二次优化的sft模型 1epoch bs16)
            # glm-4-flash:499254306::b3haz1h2 (第二次优化的sft模型 2epoch bs16)
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            top_p=0,
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)



def get_zhipu_response(prompt):
    client = ZhipuAI(api_key="0423d9de045f1539087b88fc9367e0ba.ZJlRZY4RxsTrCME9")
    try:
        response = client.chat.completions.create(
            model="glm-4-flash:499254306::auoms6xs",  #glm-4-flash, glm-4-plus,
            # glm-4-9b:499254306::9rjsdakn (第一次的sft模型 1epoch bs8)
            # glm-4-flash:499254306::mr2ahrnw (第一次的sft模型 1epoch bs8)
            # glm-4-flash:499254306::auoms6xs (第二次优化的sft模型 1epoch bs16)
            # glm-4-flash:499254306::b3haz1h2 (第二次优化的sft模型 2epoch bs16)
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            top_p=0,
            temperature=0,
            max_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)



def get_embedding(prompt):
    # 设置 API 密钥
    os.environ["OPENAI_API_KEY"] = "sk-eUazPxq20iyLu9W0A63eE1Ff71Eb4b0885D9D88cF3Ff2204"
    os.environ["OPENAI_API_BASE"] = "https://4.0.wokaai.com/v1"

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE")
    )
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=prompt
        )
        # 检查返回格式并获取嵌入
        embedding = response.data[0].embedding  # 注意这里使用点操作符
        return embedding
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    prompt = """
"""
    res = get_gpt_response(prompt)
    print(res)
