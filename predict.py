import os
import time
import math
import argparse
import jsonlines
import copy
from tqdm import tqdm

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import LinearDecayWithWarmup
from paddle.optimizer import AdamW

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import UNIMOLMHeadModel, UNIMOTokenizer, BasicTokenizer
from paddlenlp.metrics import BLEU

from gen_utils import print_args, set_seed, create_data_loader, select_sum


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset_name', type=str, default='dureader_qg', help='The name of the dataset to load.')
    parser.add_argument('--model_name_or_path', type=str, default='unimo-text-1.0', help='The path or shortcut name of the pre-trained model.')
    parser.add_argument("--predict_file", type=str, required=False, default=None, help="Predict data path.")
    parser.add_argument("--vocab_file", type=str, required=True, help="Vocab file path.")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='The directory where the checkpoints will be saved.')
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every X updates steps.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for initialization.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--max_seq_len', type=int, default=512, help='The maximum sequence length of training.')
    parser.add_argument('--max_target_len', type=int, default=30, help='The maximum target sequence length of training.')
    parser.add_argument('--max_title_len', type=int, default=30, help='The maximum title sequence length of training.')
    parser.add_argument('--max_dec_len', type=int, default=20, help='The maximum sequence length of decoding.')
    parser.add_argument('--min_dec_len', type=int, default=3, help='The minimal sequence length of decoding.')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='The numbers of returned sequences for one input in generation.')
    parser.add_argument('--decode_strategy', type=str, default='beam_search', help='The decode strategy in generation.')
    parser.add_argument('--top_k', type=int, default=0, help='The number of highest probability vocabulary tokens to keep for top-k sampling.')
    parser.add_argument('--temperature', type=float, default=1.0, help='The value used to module the next token probabilities.')
    parser.add_argument('--top_p', type=float, default=1.0, help='The cumulative probability for top-p sampling.')
    parser.add_argument('--num_beams', type=int, default=4, help='The number of beams for beam search.')
    parser.add_argument('--length_penalty', type=float, default=1.2, help='The exponential penalty to the sequence length for beam search.')
    parser.add_argument('--device', type=str, default='gpu', help='The device to select for training the model.')
    parser.add_argument('--output_path', type=str, default='./predict.txt', help='The file path where the infer result will be saved.')
    parser.add_argument("--do_predict", action='store_true', help="Whether to eval and predict.")
    args = parser.parse_args()
    return args
# yapf: enable


def read_file(file):
    with jsonlines.open(file, "r") as f:
        data = [line for line in f]
        for line in data:
            if not line:
                continue
            yield line


def run(args):
    paddle.set_device(args.device)
    world_size = dist.get_world_size()

    if world_size > 1:
        dist.init_parallel_env()
    set_seed(args.seed)

    model = UNIMOLMHeadModel.from_pretrained(args.model_name_or_path)
    tokenizer = UNIMOTokenizer(args.vocab_file)
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "[USER1]", "[USER2]",
            "[SPC0]", "[SPC1]", "[SPC2]",
            "[SIM0]", "[SIM1]", "[SIM2]",
            "[SEN0]", "[SEN1]", "[SEN2]",
            "[LEN0]", "[LEN1]", "[LEN2]",
            "[ASK0]", "[ASK1]"
        ]
    })

    if world_size > 1:
        model = paddle.DataParallel(model)

    test_ds = load_dataset(read_file, file=args.predict_file, lazy=False)
    test_ds, test_data_loader = create_data_loader(test_ds, tokenizer, args, "test")

    if args.do_predict:
        model_eval = model._layers if isinstance(model, paddle.DataParallel) else model
        prediction(model_eval, test_data_loader, args, tokenizer)


@paddle.no_grad()
def prediction(model, data_loader, args, tokenizer):
    print("\nPred begin...")
    model.eval()
    pred_ref = []
    time_begin = time.time()
    for step, inputs in enumerate(tqdm(data_loader, total=len(data_loader)), 1):
        input_ids, token_type_ids, position_ids, attention_mask, *_ = inputs
        ids, scores = model.generate(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            max_length=args.max_dec_len,
            min_length=args.min_dec_len,
            decode_strategy=args.decode_strategy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            num_return_sequences=args.num_return_sequences,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.mask_token_id,
        )

        results = select_sum(ids, scores, tokenizer, args.max_dec_len, args.num_return_sequences)
        pred_ref.extend(results)

    print("Generation cost time:", time.time() - time_begin)

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for ref in pred_ref:
            fout.write(ref + "\n")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    run(args)
