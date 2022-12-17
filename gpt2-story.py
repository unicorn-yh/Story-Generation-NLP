import os, time
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, PreTrainedTokenizer, GPT2LMHeadModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pandas as pd 


train_size = 2000
test_size = 500

def load_data():
    print('Loading data')
    train_stories = pd.read_csv("story_generation_dataset/ROCStories_train.csv", encoding="utf8")
    test_stories = pd.read_csv("story_generation_dataset/ROCStories_test.csv", encoding="utf8")
    val_stories = pd.read_csv("story_generation_dataset/ROCStories_val.csv", encoding="utf8")
    train_stories = train_stories.append(val_stories)
    train_stories = train_stories[:train_size]
    test_stories = train_stories[:test_size]
    return train_stories, test_stories

def generate_raw_text_files(train_stories,test_stories):
    index = 1
    for data in train_stories.values[:,1:]:
        with open("data/train/raw_text/"+str(index)+".txt","w") as file:
            for sent in data:
                file.write(sent+ r'\r\n\ '[:-1])
        file.close()
        index += 1

    index = 1
    for data in test_stories.values[:,1:]:
        with open("data/test/raw_text/"+str(index)+".txt","w") as file:
            for sent in data:
                file.write(sent+ r'\r\n\ '[:-1])
        file.close()
        index += 1


def tokenize_longform_text(raw_text_paths, tokenizer, block_size, drop_last=True, overlap=True):
    """ 从文本文件路径列表中加载原始 LONGFORM 文本，对其进行标记，将标记化文本拆分为训练示例并返回列表。
    需要传入 HuggingFace Transformers 预训练标记器"""

    block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
    for text_file in raw_text_paths:
        assert os.path.isfile(text_file), "{} is not a file".format(text_file)
    examples = []
    for text_file in raw_text_paths:
        with open(text_file, encoding="utf-8") as f:
            try:
                text = f.read()
                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                print("{} successfully read and tokenized".format(text_file))
            except:
                print("Error reading or tokenizing file {}".format(text_file))

        len_tokens = len(tokenized_text)
        print(len_tokens)
        if len_tokens < block_size:
            print("File {} is too short for the block size".format(text_file))
            pass
        try:
            if overlap is False:
                for i in range(0, len_tokens - block_size + 1, block_size):  # don't overlap examples
                    examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size]))
            else:  # overlap examples
                for i in range(0, len_tokens - block_size + 1, int(block_size / 2)):
                    examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size]))
            if drop_last is False:
                examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[-block_size:]))
            print("Successfully split tokens from file {} into examples".format(text_file))
        except:
            print("Failed at splitting tokens from file {} into examples".format(text_file))
    print("{} examples total tokenized".format(len(examples)))
    return examples


def make_tokenized_examples(tokenizer, block_size, root_dir_path, examples_file=None):
    """标记文本目录，其中原始文本位于子目录'/raw_text'中，默认情况下将标记化文本放入子目录'/tokenized_examples'中"""

    assert os.path.isdir(root_dir_path), "{} is not a directory".format(root_dir_path)
    # and check that we didn't accidentally put a / at the end of the dir path
    if not os.path.split(root_dir_path)[1]:
        root_dir_path = os.path.split(root_dir_path)[0]

    raw_dir_path = os.path.join(root_dir_path, "raw_text")
    assert os.path.isdir(raw_dir_path), "{} has no raw_text/ subdirectory".format(root_dir_path)

    file_list = []
    for file in os.listdir(raw_dir_path):
        if file.endswith(".txt"):
            file_list.append(os.path.join(raw_dir_path, file))
    if len(file_list) == 0:
        raise RuntimeError("No text files found in {}".format(raw_dir_path))

    if examples_file is None:
        tokenized_dir = os.path.join(root_dir_path, "tokenized_examples")
        if not os.path.isdir(tokenized_dir):
            os.mkdir(tokenized_dir)

        print(root_dir_path)
        print(os.path.split(root_dir_path))

        author_name = os.path.split(root_dir_path)[1]
        examples_file = "examples_gpt2_blocksize_{}_{}.pkl".format(block_size, author_name)
        examples_file = os.path.join(tokenized_dir, examples_file)
    else:
        assert type(examples_file) is str, "tokenized_file_name must be a string or None"
        tokenized_dir = os.path.split(examples_file)[0]
        assert os.path.isdir(tokenized_dir), "{} is not a directory".format(tokenized_dir)

    # 标记所有文件，将它们拆分为示例并连接它们
    examples = tokenize_longform_text(file_list, tokenizer, block_size, drop_last=False, overlap=True)

    # c存储为 pickle
    with open(examples_file, 'wb') as f:
        pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("{} examples created and saved in {}".format(len(examples), examples_file))

def tokenize_data():
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # 使用 gpt2 Transformer 库中的 tokenizer
    train_path = 'data/train/'
    test_path = 'data/test/'
    make_tokenized_examples(gpt2_tokenizer,10, train_path, examples_file=None)
    return gpt2_tokenizer

class StoryData(torch.utils.data.Dataset):
    '''从文件路径列表中加载标记化 gpt2 示例'''
    def __init__(self, file_paths):
        for fpath in file_paths:
            assert os.path.isfile(fpath), "{} does not exist".format(fpath)
        self.examples = []
        for fpath in file_paths:
            with open(fpath, 'rb') as f:
                examps = pickle.load(f)
            self.examples.extend(examps)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)  
    

def generate_test_text(gpt2_tokenizer, model, max_length=256, input_text=None):
    model.eval()
    if input_text is None:
        input_text = "Once upon a time there was a huge beanstalk."
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors='pt')
    input_ids = input_ids.to('cuda')
    output_ids = model.generate(input_ids, 
                                pad_token_id=gpt2_tokenizer.eos_token_id,
                                max_length=max_length, 
                                do_sample=True, 
                                top_p=0.95, 
                                top_k=60,
                                num_return_sequences=1)

    output_text = gpt2_tokenizer.decode(output_ids[0])
    return output_text


def train():
    file_path = []
    file_path.append(os.path.join('data/train/', "tokenized_examples/examples_gpt2_blocksize_10_train.pkl"))
    story_dataset = StoryData(file_path)
    story_dataloader = DataLoader(story_dataset, batch_size=1, shuffle = True)
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')  # 选择模型 distilgpt2 来训练

    # 设置超参数
    N_EPOCHS = 5
    BATCH_SIZE = 8    # 由于模型太庞大，因此使用梯度积累
    LEARNING_RATE = 0.0001  #0.00002
    WARMUP_STEPS = 100      # 10000

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps=WARMUP_STEPS, 
                                                num_training_steps=-1)  
    # 定义 scheduler（随着时间的变化来改变学习率）
    # 梯度累积文档的参考链接
    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation# 

    epoch_loss = 0
    internal_batch_count = 0  # 由于梯度积累的原因，需要一个变量用于跟踪每个批次中的示例数   
    scaler = torch.cuda.amp.GradScaler()  # 使用 FP16 缩放器来加速训练
    model.train()

    for epoch in range(N_EPOCHS): 
        print("started epoch {}".format(epoch))
        for idx, text in enumerate(story_dataloader):  
            with torch.cuda.amp.autocast():  # 前向传播
                outputs = model(text.to(device), labels=(text.to(device)))
                loss, logits = outputs[:2]  # loss 损失用于反向传播
                loss = loss / BATCH_SIZE 
            scaler.scale(loss).backward()
            epoch_loss = epoch_loss + loss.detach().cpu().numpy()  
            internal_batch_count = internal_batch_count + 1
            if internal_batch_count == BATCH_SIZE:
                internal_batch_count = 0 
                scaler.step(optimizer) 
                scaler.update()
                optimizer.zero_grad() # 将优化器中的梯度归零
                model.zero_grad() # 将我们在模型中累积的梯度归零
                scheduler.step() # 执行调度步骤
        model.eval()
        print("Epoch {} has loss {}".format(epoch, epoch_loss))
        epoch_loss = 0
        model.train()
    return model
    
def strip_paragraph(paragraph, sentence=6):
    sent_count = 0
    output_str = ""
    for i in range(len(paragraph)):
        output_str += paragraph[i]
        if paragraph[i] == '.' and paragraph[i+1] == ' ':
            sent_count += 1
        if sent_count == sentence:
            break
    return output_str

def generate_story(gpt2_tokenizer, model,train_stories):
    test_ls = np.loadtxt("test-index.txt")
    train_array = train_stories.values[:,1:].reshape(-1).tolist()
    with open("output/gpt2-generated-story.txt","w") as file:
        for i in range(20):
            output_text = ""
            index = int(test_ls[i])
            prompt = train_array[index] 
            text = str(generate_test_text(gpt2_tokenizer, model,input_text=prompt, max_length=256)).replace('\\','').replace('rn',' ')
            text = strip_paragraph(text,sentence=6)
            output_text += "Original story: " + prompt + "\n"
            output_text += "Generated story: " + text + "\n\n"
            file.write(output_text)
            print(output_text)
    file.close()

if __name__ == "__main__":
    train_stories, test_stories = load_data()
    generate_raw_text_files(train_stories,test_stories)
    gpt2_tokenizer = tokenize_data()
    model = train()
    generate_story(gpt2_tokenizer, model, train_stories)





