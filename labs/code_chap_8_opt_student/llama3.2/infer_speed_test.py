import transformers
import torch
import torch_mlu
import time

model_id = "/workspace/model/favorite/large-scale-models/model-v1/Llama-3.2-3B/"

#TODO: 创建一个文本生成的管道，指定任务类型、模型路径、数据类型，并在MLU设备上运行
pipeline = transformers.pipeline(
    task="text-generation",
    model=model_id,
    device=torch.mlu.current_device(),
    torch_dtype=torch.float16
)

messages = [
    {"role": "system", "content": "You are a story writing chatbot"},
    {"role": "user", "content": "Once upon a time, .... start to write a very long story"},
]

#TODO: 应用聊天模板，将消息转化为适合模型的输入格式
prompt = "<|system|>\nYou are a story writing chatbot\n<|user|>\nOnce upon a time, .... start to write a very long story\n<|assistant|>\n"
#TODO: 定义文本生成的终止符列表，包含模型的结束标记和自定义标记<|eot_id|> 对应的 token ID。
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
terminators = [tokenizer.eos_token_id]

# 初始化一个空列表，用于存储每次迭代中计算出的每秒生成的tokens数量
times = []
for i in range(1):
    max_length = 256
    # print("========================")
    # print("Iteration", i)
    # print("========================")
    #TODO: 记录开始时间，用于计算生成任务所需的时间
    start_time = time.time()
    #TODO: 调用文本生成管道，根据给定的 prompt 和参数生成文本
    outputs = pipeline(
        prompt,
        max_new_tokens=max_length,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    # print(outputs[0]["generated_text"][len(prompt):])
    #TODO: 记录生成结束时间
    end_time = time.time()
    #TODO: 计算本次生成任务的耗时
    elapsed_time = end_time - start_time
    #TODO: 计算每秒生成的 tokens 数量，作为本次生成任务的吞吐量
    tokens_per_sec = max_length / elapsed_time if elapsed_time > 0 else 0
    #TODO: 将计算出的每秒生成的 tokens 数量存入 times 列表
    times.append(tokens_per_sec)
    print(f"iter: {i}, Tokens per second: {tokens_per_sec}")

print("========================")
#TODO: 计算并打印平均每秒生成的 tokens 数量
print("Average tokens per second:",  sum(times) / len(times) if times else 0)
print("========================")

print("INFERSPEED PASS!")
