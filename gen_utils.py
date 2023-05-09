import random
from functools import partial

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
from paddlenlp.data import Pad


def print_args(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better.
    paddle.seed(seed + dist.get_rank())


def tokenizer_encode(source, target, tokenizer, max_seq_len, max_target_len):

    # encode target sentence
    target_ids = []
    if target is not None:
        target_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(target.split(" "))
        if len(target_ids) > max_target_len - 1:
            target_ids = target_ids[: max_target_len - 1]

        # add `mask_token` as the eos token
        target_ids += [tokenizer.mask_token_id]
    else:
        target_ids = [tokenizer.cls_token_id]

    # encode source sentence
    max_source_len = max_seq_len - len(target_ids)
    source_ids = []
    source_ids = tokenizer.convert_tokens_to_ids(source.split(" "))
    if len(source_ids) > max_source_len - 1:
        source_ids = source_ids[: max_source_len - 1]
    source_ids += [tokenizer.sep_token_id]

    encoded_inputs = {}
    encoded_inputs["input_ids"] = source_ids + target_ids
    sequence_length = len(encoded_inputs["input_ids"])
    assert sequence_length <= max_seq_len
    encoded_inputs["seq_len"] = sequence_length
    encoded_inputs["token_type_ids"] = [0] * len(source_ids) + [1] * len(target_ids)

    encoded_inputs["position_ids"] = list(range(len(source_ids))) + list(range(len(target_ids)))
    attention_mask = np.ones((sequence_length, sequence_length), dtype="float32") * -1e4
    start = len(source_ids)
    end = sequence_length
    attention_mask[:end, :start] = 0.0
    # Generate the lower triangular matrix using the slice of matrix
    tmp = np.triu(np.ones([end - start, end - start], dtype="float32") * -1e4, 1)
    attention_mask[start:end, start:end] = tmp
    encoded_inputs["attention_mask"] = attention_mask
    
    return encoded_inputs


def convert_example(
    example, tokenizer, max_seq_len=512, max_target_len=128, max_title_len=256, mode="train"
):
    """Convert all examples into necessary features."""
    
    source = example["source"]
    target = example["target"]

    tokenized_example = tokenizer_encode(
        source,
        target=target if mode != 'test' else None,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_target_len=max_target_len,
    )

    if mode != 'test':
        index_list = []
        count = tokenized_example["input_ids"].count(tokenizer.cls_token_id)
        index = -1
        for i in range(0, count):
            index = tokenized_example["input_ids"].index(tokenizer.cls_token_id, index + 1)
            index_list.append(index)
        target_start = index_list[-1]
        target_end = tokenized_example["seq_len"]
        # Use to gather the logits corresponding to the labels during training
        tokenized_example["masked_positions"] = list(range(target_start, target_end - 1))
        tokenized_example["labels"] = tokenized_example["input_ids"][target_start + 1 : target_end]
    else:
        tokenized_example["target"] = example["target"]

    return tokenized_example


def batchify_fn(batch_examples, pad_val, mode):
    def pad_mask(batch_attention_mask):
        batch_size = len(batch_attention_mask)
        max_len = max(map(len, batch_attention_mask))
        attention_mask = np.ones((batch_size, max_len, max_len), dtype="float32") * -1e9
        for i, mask_data in enumerate(attention_mask):
            seq_len = len(batch_attention_mask[i])
            mask_data[-seq_len:, -seq_len:] = np.array(batch_attention_mask[i], dtype="float32")
        # In order to ensure the correct broadcasting mechanism, expand one
        # dimension to the second dimension (n_head of Transformer).
        attention_mask = np.expand_dims(attention_mask, axis=1)
        return attention_mask

    pad_func = Pad(pad_val=pad_val, pad_right=False, dtype="int64")

    input_ids = pad_func([example["input_ids"] for example in batch_examples])
    token_type_ids = pad_func([example["token_type_ids"] for example in batch_examples])
    position_ids = pad_func([example["position_ids"] for example in batch_examples])

    attention_mask = pad_mask([example["attention_mask"] for example in batch_examples])

    """
    @mode: 
        1. @mode in ["train", "dev"]: return all elements for calculating perplexity;
        2. @mode is "test": return partial elements for calculating bleu-4 scores.
    """
    if mode != 'test':
        max_len = max([example["seq_len"] for example in batch_examples])
        masked_positions = np.concatenate(
            [
                np.array(example["masked_positions"]) + (max_len - example["seq_len"]) + i * max_len
                for i, example in enumerate(batch_examples)
            ]
        )
        labels = np.concatenate([np.array(example["labels"], dtype="int64") for example in batch_examples])
        return input_ids, token_type_ids, position_ids, attention_mask, masked_positions, labels
    else:
        return input_ids, token_type_ids, position_ids, attention_mask
    

def create_data_loader(dataset, tokenizer, args, mode):
    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        max_target_len=args.max_target_len,
        max_title_len=args.max_title_len,
        mode=mode,
    )
    dataset = dataset.map(trans_func, lazy=True)

    if mode == "train":
        batch_sampler = DistributedBatchSampler(dataset, batch_size=args.batch_size, shuffle=True)
    elif mode == "dev" or mode == "test":
        batch_sampler = BatchSampler(dataset, batch_size=args.batch_size, shuffle=False)
    collate_fn = partial(batchify_fn, pad_val=tokenizer.pad_token_id, mode=mode)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, return_list=True)
    return dataset, data_loader


def post_process_sum(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.mask_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    special_tokens = ["[UNK]"]
    tokens = [token for token in tokens if token not in special_tokens]
    return token_ids, tokens


def remove_template(instr):
    """Remove template prefix of decoded sequence."""
    outstr = instr[len("[USER1] response : "):]
    return outstr


def select_sum(ids, scores, tokenizer, max_dec_len=None, num_return_sequences=1):
    results = []
    group = []
    tmp = []
    if scores is not None:
        ids = ids.numpy()
        scores = scores.numpy()

        if len(ids) != len(scores) or (len(ids) % num_return_sequences) != 0:
            raise ValueError(
                "the length of `ids` is {}, but the `num_return_sequences` is {}".format(
                    len(ids), num_return_sequences
                )
            )

        for pred, score in zip(ids, scores):
            pred_token_ids, pred_tokens = post_process_sum(pred, tokenizer)
            num_token = len(pred_token_ids)

            target = " ".join(pred_tokens)
            target = remove_template(target)

            # not ending
            if max_dec_len is not None and num_token >= max_dec_len:
                score -= 1e3

            tmp.append([target, score])
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []

        for preds in group:
            preds = sorted(preds, key=lambda x: -x[1])
            results.append(preds[0][0])
    else:
        ids = ids.numpy()

        for pred in ids:
            pred_token_ids, pred_tokens = post_process_sum(pred, tokenizer)
            num_token = len(pred_token_ids)
            response = " ".join(pred_tokens)
            response = remove_template(response)

            # TODO: Support return scores in FT.
            tmp.append([response])
            if len(tmp) == num_return_sequences:
                group.append(tmp)
                tmp = []

        for preds in group:
            results.append(preds[0][0])

    return results
