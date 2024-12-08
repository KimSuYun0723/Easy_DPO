from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda:1" if torch.cuda.is_available() else "cpu"

print("==== Llama-3.2B Vanila ====")
base_model = AutoModelForCausalLM.from_pretrained("Bllossom/llama-3.2-Korean-Bllossom-3B").to(device)
base_tokenizer = AutoTokenizer.from_pretrained("Bllossom/llama-3.2-Korean-Bllossom-3B")
base_tokenizer.pad_token = base_tokenizer.eos_token if base_tokenizer.pad_token is None else base_tokenizer.pad_token

################### 질문1 입력
question_1 = "안녕!"
inputs = base_tokenizer(question_1, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_1 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 1:", question_1)
print("모델의 답변:", answer_1)

################### 질문2 입력
question_2 = "정국이 5위입니다. 정국보다 결승선을 먼저 통과한 사람의 수를 찾아보세요."
inputs = base_tokenizer(question_2, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_2 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 2:", question_2)
print("모델의 답변:", answer_2)
print()


################### 질문3 입력
question_3 = "너 한국어를 이해하니?"
inputs = base_tokenizer(question_3, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_3 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 3:", question_3)
print("모델의 답변:", answer_3)
print()

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("==== 영어:한국어=8:2 ====")
base_model = AutoModelForCausalLM.from_pretrained("/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/final_dir/data_82").to(device)
base_tokenizer = AutoTokenizer.from_pretrained("/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/final_dir/data_82")
base_tokenizer.pad_token = base_tokenizer.eos_token if base_tokenizer.pad_token is None else base_tokenizer.pad_token

################### 질문1 입력
question_1 = "안녕!"
inputs = base_tokenizer(question_1, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_1 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 1:", question_1)
print("모델의 답변:", answer_1)

################### 질문2 입력
question_2 = "정국이 5위입니다. 정국보다 결승선을 먼저 통과한 사람의 수를 찾아보세요."
inputs = base_tokenizer(question_2, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_2 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 2:", question_2)
print("모델의 답변:", answer_2)
print()

################### 질문3 입력
question_3 = "너 한국어를 이해하니?"
inputs = base_tokenizer(question_3, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_3 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 3:", question_3)
print("모델의 답변:", answer_3)
print()


print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("==== 영어:한국어=5:5 ====")
base_model = AutoModelForCausalLM.from_pretrained("/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/final_dir/data_55").to(device)
base_tokenizer = AutoTokenizer.from_pretrained("/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/final_dir/data_55")
base_tokenizer.pad_token = base_tokenizer.eos_token if base_tokenizer.pad_token is None else base_tokenizer.pad_token

################### 질문1 입력
question_1 = "안녕!"
inputs = base_tokenizer(question_1, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_1 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 1:", question_1)
print("모델의 답변:", answer_1)

################### 질문2 입력
question_2 = "정국이 5위입니다. 정국보다 결승선을 먼저 통과한 사람의 수를 찾아보세요."
inputs = base_tokenizer(question_2, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_2 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 2:", question_2)
print("모델의 답변:", answer_2)
print()

################### 질문3 입력
question_3 = "너 한국어를 이해하니?"
inputs = base_tokenizer(question_3, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_3 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 3:", question_3)
print("모델의 답변:", answer_3)
print()


print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("==== 영어:한국어=2:8 ====")
base_model = AutoModelForCausalLM.from_pretrained("/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/final_dir/data_28").to(device)
base_tokenizer = AutoTokenizer.from_pretrained("/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/final_dir/data_28")
base_tokenizer.pad_token = base_tokenizer.eos_token if base_tokenizer.pad_token is None else base_tokenizer.pad_token

################### 질문1 입력
question_1 = "안녕!"
inputs = base_tokenizer(question_1, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_1 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 1:", question_1)
print("모델의 답변:", answer_1)

################### 질문2 입력
question_2 = "정국이 5위입니다. 정국보다 결승선을 먼저 통과한 사람의 수를 찾아보세요."
inputs = base_tokenizer(question_2, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_2 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 2:", question_2)
print("모델의 답변:", answer_2)
print()

################### 질문3 입력
question_3 = "너 한국어를 이해하니?"
inputs = base_tokenizer(question_3, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_3 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 3:", question_3)
print("모델의 답변:", answer_3)
print()

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("==== 한국어 100 ====")
base_model = AutoModelForCausalLM.from_pretrained("/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/final_dir/data_ko").to(device)
base_tokenizer = AutoTokenizer.from_pretrained("/home/nlpgpu8/hdd2/suyun/DPO_practice/easy_DPO/final_dir/data_ko")
base_tokenizer.pad_token = base_tokenizer.eos_token if base_tokenizer.pad_token is None else base_tokenizer.pad_token

################### 질문1 입력
question_1 = "안녕!"
inputs = base_tokenizer(question_1, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_1 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 1:", question_1)
print("모델의 답변:", answer_1)

################### 질문2 입력
question_2 = "정국이 5위입니다. 정국보다 결승선을 먼저 통과한 사람의 수를 찾아보세요."
inputs = base_tokenizer(question_2, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_2 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 2:", question_2)
print("모델의 답변:", answer_2)

################### 질문3 입력
question_3 = "너 한국어를 이해하니?"
inputs = base_tokenizer(question_3, return_tensors="pt", padding=True).to(device)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = base_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=300,  # 더 길게 설정
    temperature=0.1,  # 낮은 값으로 설정
    num_beams=5,
    pad_token_id=base_tokenizer.pad_token_id
)
answer_3 = base_tokenizer.decode(output[0], skip_special_tokens=True)
print("질문 3:", question_3)
print("모델의 답변:", answer_3)
print()