# 可控对话生成

**目录**
- [可控对话生成](#可控对话生成)
  - [简介](#简介)
  - [训练定制](#训练定制)
    - [环境依赖](#环境依赖)
    - [代码结构说明](#代码结构说明)
    - [可控对话生成训练全流程介绍](#可控对话生成训练全流程介绍)
    - [数据准备](#数据准备)
      - [数据加载](#数据加载)
      - [数据处理](#数据处理)
      - [从本地文件创建数据集-可选](#从本地文件创建数据集-可选)
    - [模型训练](#模型训练)
    - [模型预测](#模型预测)
  - [References](#references)

## 简介
Controllable Dialogue Generation（CDG），即可控对话生成，指的是给定一段上下文和指定的属性，自动生成一个流畅、符合上下文且满足给定属性要求的回复。

可控对话生成技术在教育、咨询等多个领域均有着巨大的应用价值。具体来说，可控对话生成可广泛应用于问答系统，对话提问，聊天机器人，闲聊机器人主动提问等等场景。

本项目是基于预训练语言模型UNIMO-Text的可控对话生成，具有以下优势：

- 效果领先。基于百度自研中文预训练语言模型UNIMO-Text。
- 高性能推理。本项目基于FasterTransformer进行推理加速，能够提供更高性能的推理体验，优化后的推理模型在dureader_qg开发集的推理耗时缩短为优化前的1/5。
- 训练推理部署全流程打通。本项目提供了全面的定制训练流程，从数据准备、模型训练预测，到模型推理部署，一应俱全。

## 训练定制

### 环境依赖
- nltk
- evaluate
- tqdm
- jsonlines

安装方式：`pip install -r requirements.txt`

### 代码结构说明

以下是本项目主要代码结构及说明：

```text
├── data # 数据
│   ├── train.jsonl # 训练数据
│   ├── dev.jsonl # 验证数据
│   └── test.jsonl # 测试数据
├── persona # persona 任务配置
│   ├── config.json # 模型结构配置文件
│   └── vocab.txt # 词表
├── train.py # 训练代码
├── predict.py # 预测评估代码
├── gen_utils.py # 工具函数代码
├── train.sh # 训练脚本
├── inference.sh # 推理脚本
└── README.md # 说明文档
```

### 可控对话生成定制训练全流程介绍
接下来，我们将按数据准备、训练、预测、推理部署等四个阶段对问题生成应用的全流程进行介绍。
1. **数据准备**
- 使用已做好属性处理的对话生成数据集Persona-Chat进行实验。

2. **模型训练**

- 数据准备完成后，可以开始使用我们的数据集对模型进行训练。首先根据任务需求，调整可配置参数，选择使用GPU或CPU进行模型训练，脚本默认保存在开发集最佳表现模型。


3. **模型预测**

- 训练结束后，我们可以加载保存的最佳模型进行模型测试，打印模型预测结果。

### 数据准备
#### 数据加载
[**Persona-Chat**数据集]是一个英文个性化对话生成生成数据集，我们使用该数据集作为应用案例进行实验。**Persona-Chat**中的数据主要由用户描述、对话历史、回复3个主要部分组成。用户描述是使用多个句子刻画用户形象；对话历史则是由多轮对话组成。为研究可控对话生成任务，我们在Persona-Chat数据集的基础上，基于统计和评估等方式构造了5个属性，属性介绍与预处理细节如下：
- 特异性（Specificity）：统计回复中各词词频，经归一化后离散为3类标签；
- 情感（Sentiment）：利用Stanford CoreNLP对回复中的情感进行标注，标签为{0:position, 1:neutral, 2:negative};
- 回复相关性（Response-relatedness)：基于余弦相似度计算回复与上文的相关性（基于Glove embedding计算）;
- 是否为问句（Question-asking）：根据关键词对回复的句式进行简单标注。关键词为：{how, what, when, where, which, who, whom, whose, why, ?};
- 长度（Length）：统计回复长度，并离散为3类标签。


#### 数据处理
针对**Persona-Chat**数据集，我们需要将可控对话生成任务格式的数据进行转换从而得到text2text形式的数据，我们基于以下模型构造数据：
```text
# source
persona: <persona_text> context: <context_text> attributes: <attributes_list>

# target
Response: <response_text>
```

处理好的数据集文件格式如下：
- train.jsonl/dev.jsonl/test.jsonl 文件格式：
```text
{
  "source": <source_text>,
  "target": <target_text>,
}
```

### 模型训练
运行如下命令即可在样例训练集上进行训练，并在样例验证集上进行验证。
```shell
# GPU启动，参数`--gpus`指定训练所用的GPU卡号，可以是单卡，也可以多卡
# 例如使用1号和2号卡，则：`--gpu 1,2`
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0,1" --log_dir ./save/persona/log train.py \
    --model_name_or_path="./persona/" \
    --vocab_file=./persona/vocab.txt \
    --save_dir=./save/persona/checkpoints \
    --output_path=./save/persona/predict.txt \
    --logging_steps=100 \
    --save_steps=500 \
    --epochs=30 \
    --batch_size=128 \
    --learning_rate=5e-4 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_eval \
    --max_dec_len=30 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --device=gpu \
    --train_file data/train.jsonl \
    --predict_file data/dev.jsonl
```

关键参数释义如下：
- `gpus` 指示了训练所用的GPU，使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"。
- `vocab_file` 词表文件地址。
- `train_file` 本地训练数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为None。
- `predict_file` 本地测试数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为None。
- `save_dir` 表示模型的保存路径。
- `output_path` 表示预测结果的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `epochs` 表示训练轮数。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `warmup_propotion` 表示学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数占总步数的比例。
- `max_seq_len` 模型输入序列的最大长度。
- `max_target_len` 模型训练时标签的最大长度。
- `min_dec_len` 模型生成序列的最小长度。
- `max_dec_len` 模型生成序列的最大长度。
- `do_train` 是否进行训练。
- `do_predict` 是否进行预测，在验证集上会自动评估。
- `device` 表示使用的设备，从gpu和cpu中选择。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中。如：

```text
./unimo/finetune/checkpoints
├── model_1000
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。

微调的模型在dureader_qg验证集上有如下结果(指标为BLEU-4)，其中`unimo-text-1.0-dureader_qg-w/o-template`表示不使用模版策略微调的结果，`unimo-text-1.0-large-dureader_qg`表示使用large模型微调的结果，`unimo-text-1.0-question-generation-dureader_qg`表示在通用问题生成预训练模型`unimo-text-1.0-question-generation`上微调的结果：

|       model_name        | DuReaderQG |
| :-----------------------------: | :-----------: |
|    unimo-text-1.0-dureader_qg-w/o-template    | 39.61 |
|    unimo-text-1.0-dureader_qg    | 41.08 |
|    unimo-text-1.0-large-dureader_qg    | 41.51 |
|    unimo-text-1.0-question-generation-dureader_qg    | 44.02 |

### 模型预测

运行下方脚本可以使用训练好的模型进行预测。

```shell
export CUDA_VISIBLE_DEVICES=0
python -u predict.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=your_model_path \
    --output_path=./predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --template=1 \
    --device=gpu
```
关键参数释义如下：
- `output_path` 表示预测输出结果保存的文件路径，默认为./predict.txt。
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的微调好的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。

## References
Hu, Zhe, et al. "Controllable Dialogue Generation with Disentangled Multi-grained Style Specification and Attribute Consistency Reward." TASLP2022.
