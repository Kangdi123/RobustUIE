import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


def build_prompt(query, resp='', system=None, sep='\n'):
    if system is None:
        system = (
            "You are a highly skilled assistant at digesting "
            "and extracting information from textual content. "
            "Below is an input containing standard type definitions "
            "and textual content. Please complete it with the "
            "extracted information in the form of structured code."
        )
    pattern = "Input:\n{query}\nOutput:\n{resp}"
    if system:
        return sep.join([system, pattern.format(query=query, resp=resp)])
    else:
        return pattern.format(query=query, resp=resp)


def build_input(tokenizer, query, resp, system=None, sep='\n', head_max_len=1500, tail_max_len=2000):
    prompt = build_prompt(query=query, resp=resp, system=system, sep=sep)
    token_ids = tokenizer.encode(prompt)
    if head_max_len is not None and tail_max_len is not None and head_max_len + tail_max_len < len(token_ids):
        token_ids = token_ids[:head_max_len] + token_ids[-tail_max_len:]
    return token_ids


def calculate_loss(data, tokenizer, model, device, head_max_len=1500, tail_max_len=2000):
    """
    计算每个样本的损失并返回带损失的样本列表。

    参数:
        data (list): 输入的数据列表，每个元素包含一个字典，包含 'one-stage' 和 'output' 信息。
        tokenizer (Tokenizer): 用于编码文本的分词器。
        model (torch.nn.Module): 用于计算损失的模型。
        device (torch.device): 模型的计算设备，通常是 'cuda' 或 'cpu'。
        head_max_len (int): 输入文本头部的最大长度，默认为1500。
        tail_max_len (int): 输入文本尾部的最大长度，默认为2000。

    返回:
        list: 包含计算损失后的样本列表，每个样本包含损失值。
    """
    samples_with_loss = []

    for item in data:
        with torch.no_grad():  # 确保不计算梯度
            # 编码输入和目标文本
            input_text = str(item['one-stage']['zero-shot']['prompt'])
            target_text = str(item['one-stage']['output'])

            # 编码输入文本
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)

            # 编码目标文本
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)

            # 构建完整的输入序列
            full_input_ids = build_input(tokenizer, input_text, target_text, head_max_len=head_max_len,
                                         tail_max_len=tail_max_len)

            # 将 full_input_ids 转换为 PyTorch 张量并移动到GPU上
            full_input_ids = torch.tensor([full_input_ids]).to(device)

            # 创建 labels，目标文本部分为 target_ids，其余部分为 -100（忽略损失）
            labels = torch.full_like(full_input_ids, -100).to(device)
            labels[0, -len(target_ids):] = torch.tensor(target_ids).to(device)

            # 计算模型输出
            outputs = model(input_ids=full_input_ids, labels=labels)
            loss = outputs.loss.item()

            # 输出损失
            print(f"损失 (Loss) for sample: {loss}")

            item['loss'] = loss
            samples_with_loss.append(item)

    return samples_with_loss


def filter_data_by_loss(data, dataset_name, noise_type, top_percentage):
    """
    根据数据集名称和噪声类型过滤数据，并返回高损失样本。
    """
    filtered_data = [item for item in data if item['dataset'] == dataset_name]
    filtered_data.sort(key=lambda x: x['loss'], reverse=True)
    top_n = int(len(filtered_data) * top_percentage)
    return filtered_data[:top_n]


# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 KnowCoder 模型和 tokenizer
model_name = "KnowCoder"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 将模型移动到GPU上
model.to(device)

# 设置模型为评估模式以节省内存并避免梯度计算
model.eval()

top_percentage = 0.3

# 定义数据集与噪声类型的对应列表
configurations = [
    ('ACE2005-ED', 'Typo Injection'),
    ('ACE2005-ED', 'Lowercase Conversion'),
    ('ACE2005-ED', 'Extend Sentence'),
    ('ACE2005-ED', 'Replace Event'),
    ('ACE2005-ED', 'Mask Context'),
    ('ACE2005-NER', 'Typo Injection'),
    ('ACE2005-NER', 'Lowercase Conversion'),
    ('ACE2005-NER', 'Extend Sentence'),
    ('ACE2005-NER', 'Replace Entity'),
    ('ACE2005-NER', 'Mask Context'),
    ('Conll2003', 'Typo Injection'),
    ('Conll2003', 'Lowercase Conversion'),
    ('Conll2003', 'Extend Sentence'),
    ('Conll2003', 'Replace Entity'),
    ('Conll2003', 'Mask Context'),
    ('WikiANN', 'Typo Injection'),
    ('WikiANN', 'Lowercase Conversion'),
    ('WikiANN', 'Extend Sentence'),
    ('WikiANN', 'Replace Entity'),
    ('WikiANN', 'Mask Context'),
    ('ACE2005-RE', 'Typo Injection'),
    ('ACE2005-RE', 'Lowercase Conversion'),
    ('ACE2005-RE', 'Extend Sentence'),
    ('ACE2005-RE', 'Replace Relation'),
    ('NYT', 'Typo Injection'),
    ('NYT', 'Lowercase Conversion'),
    ('NYT', 'Extend Sentence'),
    ('NYT', 'Replace Relation'),
]

# 加载数据
with open('augment_data.json', 'r', encoding='utf-8') as input_file:
    data = [json.loads(line) for line in input_file]

samples_with_loss = calculate_loss(data, tokenizer, model, device)

filter_sample = []

for dataset_name, noise_type in configurations:
    filter_sample = filter_data_by_loss(samples_with_loss, dataset_name, noise_type, top_percentage)
    filter_sample.extend(filter_sample)

# 将合并的数据写入文件，每条数据占一行
with open('data/filter_sample_loss.json', 'w', encoding='utf-8') as output_file:
    for item in filter_sample:
        json.dump(item, output_file, ensure_ascii=False)
        output_file.write('\n')

