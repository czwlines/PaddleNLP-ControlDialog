import os
import time
import math
import argparse
import jsonlines
import copy

import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import LinearDecayWithWarmup
from paddle.optimizer import AdamW

from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import UNIMOConfig, UNIMOLMHeadModel, UNIMOTokenizer, BasicTokenizer
from paddlenlp.metrics import BLEU

from gen_utils import print_args, set_seed, create_data_loader, select_sum


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset_name', type=str, default='dureader_qg', help='The name of the dataset to load.')
    parser.add_argument('--model_name_or_path', type=str, default='unimo-text-1.0', help='The path or shortcut name of the pre-trained model.')
    parser.add_argument("--vocab_file", type=str, required=True, help="Vocab file path.")
    parser.add_argument("--train_file", type=str, required=False, default=None, help="Train data path.")
    parser.add_argument("--predict_file", type=str, required=False, default=None, help="Predict data path.")
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='The directory where the checkpoints will be saved.')
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save checkpoint every X updates steps.')
    parser.add_argument('--predict_steps', type=int, default=2000, help='Predict checkpoint every X updates steps.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for initialization.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU/CPU for training.')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='The initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='The weight decay for optimizer.')
    parser.add_argument('--epochs', type=int, default=3, help='Total number of training epochs to perform.')
    parser.add_argument('--warmup_propotion', type=float, default=0.02, help='The number of warmup steps.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='The max value of grad norm.')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='beta2')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='epsilon')
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
    parser.add_argument("--do_train", action='store_true', help="Whether to train the model.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to eval the checkpoint by calculating perplexity.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to eval the checkpoint by predicting response and calculating bleu-4 scores.")
    args = parser.parse_args()
    return args
# yapf: enable


def calc_bleu_n(preds, targets, n_size=4):
    assert len(preds) == len(targets), (
        "The length of pred_responses should be equal to the length of "
        "target_responses. But received {} and {}.".format(len(preds), len(targets))
    )
    bleu = BLEU(n_size=n_size)
    tokenizer = BasicTokenizer()

    for pred, target in zip(preds, targets):
        pred_tokens = tokenizer.tokenize(pred)
        target_token = tokenizer.tokenize(target)

        bleu.add_inst(pred_tokens, [target_token])

    print("\n" + "*" * 15)
    print("The auto evaluation result is:")
    print("BLEU-" + str(n_size) + ":", bleu.score())
    return bleu.score()


def calc_bleu(preds, targets):
    calc_bleu_n(preds, targets, 1)
    calc_bleu_n(preds, targets, 2)
    calc_bleu_n(preds, targets, 3)
    bleu4_score = calc_bleu_n(preds, targets, 4)
    return bleu4_score


def read_file(file):
    with jsonlines.open(file, "r") as f:
        for line in f:
            if not line:
                continue
            yield line


def save_ckpt(model, tokenizer, save_dir, name):
    output_dir = os.path.join(save_dir, "model_{}".format(name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Need better way to get inner model of DataParallel
    model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def run(args):
    paddle.set_device(args.device)
    world_size = dist.get_world_size()

    if world_size > 1:
        dist.init_parallel_env()
    set_seed(args.seed)
    
    # construct model from config
    config = UNIMOConfig.from_json_file(f"{args.model_name_or_path}/config.json")
    model = UNIMOLMHeadModel(config)

    # construct tokenizer
    tokenizer = UNIMOTokenizer(args.vocab_file)
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            # the user token in dialog
            "[USER1]", "[USER2]",
            # the attribute: specificity, indicating the complexity of the token used in the response;
            "[SPC0]", "[SPC1]", "[SPC2]",
            # the attribute: similarity, indicating the the degree of relevance to above context;
            "[SIM0]", "[SIM1]", "[SIM2]",
            # the attribute: sentiment, indicating the emotional state in the response;
            "[SEN0]", "[SEN1]", "[SEN2]",
            # the attribute: length, indicating the length of the response;
            "[LEN0]", "[LEN1]", "[LEN2]",
            # the attribute: is-asking, indicating whether the response is presented in a question.
            "[ASK0]", "[ASK1]"
        ]
    })

    if world_size > 1:
        model = paddle.DataParallel(model)

    train_ds = load_dataset(read_file, file=args.train_file, lazy=False)
    dev_ds = load_dataset(read_file, file=args.predict_file, lazy=False)
    train_ds, train_data_loader = create_data_loader(train_ds, tokenizer, args, "train")
    dev_ds, dev_data_loader = create_data_loader(dev_ds, tokenizer, args, "dev")

    if args.do_train:
        num_training_steps = args.epochs * len(train_data_loader)

        lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_propotion)
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.

        decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]

        optimizer = AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            beta1=args.beta1,
            beta2=args.beta2,
            epsilon=args.epsilon,
            apply_decay_param_fun=lambda x: x in decay_params,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm),
        )

        step = 0
        total_time = 0.
        best_bleu4 = 0
        best_ppl = 1e5
        for epoch in range(args.epochs):
            print("\nEpoch %d/%d" % (epoch + 1, args.epochs))
            batch_start_time = time.time()
            for inputs in train_data_loader:
                step += 1
                labels = inputs[-1]
                logits = model(*inputs[:-1])
                labels = paddle.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
                labels = paddle.nn.functional.label_smooth(labels)
                loss = F.cross_entropy(logits, labels, soft_label=True)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

                total_time += time.time() - batch_start_time
                if step % args.logging_steps == 0:
                    ppl = paddle.exp(loss)
                    print(
                        "step %d - loss: %.4f - ppl: %.4f - lr: %.7f - %.3fs/step"
                        % (step, loss, ppl, optimizer.get_lr(), total_time / args.logging_steps)
                    )
                    total_time = 0.0

                if step % args.save_steps == 0 or step >= num_training_steps:
                    if dist.get_rank() == 0:
                        save_ckpt(model, tokenizer, args.save_dir, step)
                        print("Saved step {} model.\n".format(step))

                        if args.do_eval:
                            model_eval = model._layers if isinstance(model, paddle.DataParallel) else model
                            eval_ppl = evaluation(model_eval, dev_data_loader, args, tokenizer)
                            if eval_ppl < best_ppl:
                                print("best ppl performence has been updated: %.5f  --> %.5f" % (best_ppl, eval_ppl))
                                best_ppl = eval_ppl
                                save_ckpt(model, tokenizer, args.save_dir, "best_ppl")

                if step % args.predict_steps == 0 or step >= num_training_steps:
                    if dist.get_rank() == 0 and args.do_predict:
                        model_eval = model._layers if isinstance(model, paddle.DataParallel) else model
                        bleu4 = prediction(model_eval, dev_data_loader, args, tokenizer)
                        if bleu4 > best_bleu4:
                            print("best BLEU-4 performence has been updated: %.5f  --> %.5f" % (best_bleu4, bleu4))
                            best_bleu4 = bleu4
                            save_ckpt(model, tokenizer, args.save_dir, "best_bleu")

                batch_start_time = time.time()

        print("\nTraining completed.")

    if args.do_predict:
        model_eval = model._layers if isinstance(model, paddle.DataParallel) else model
        prediction(model_eval, dev_data_loader, args, tokenizer)


@paddle.no_grad()
def evaluation(model, data_loader, args, tokenizer):
    print("\nEval begin...")
    model.eval()
    total_loss = []
    for step, inputs in enumerate(data_loader, 1):
        assert len(inputs) == 6
        
        labels = inputs[-1]
        logits = model(*inputs[:-1])
        labels = paddle.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
        labels = paddle.nn.functional.label_smooth(labels)
        loss = F.cross_entropy(logits, labels, soft_label=True)
        total_loss.append(loss)
    
    loss = sum(total_loss) / len(total_loss)
    ppl = paddle.exp(loss)
    print(
        "Evaluation loss: %.4f - ppl: %.4f"
        % (loss, ppl)
    )
    return ppl


@paddle.no_grad()
def prediction(model, data_loader, args, tokenizer):
    print("\nPredict begin...")
    model.eval()
    pred_ref = []
    time_begin = time.time()
    total_time = 0.0
    start_time = time.time()
    for step, inputs in enumerate(data_loader, 1):
        assert len(inputs) == 6

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

        total_time += time.time() - start_time
        if step % args.logging_steps == 0:
            print("step %d - %.3fs/step" % (step, total_time / args.logging_steps))
            total_time = 0.0

        results = select_sum(ids, scores, tokenizer, args.max_dec_len, args.num_return_sequences)
        pred_ref.extend(results)
        start_time = time.time()
    print("Generation cost time:", time.time() - time_begin)

    with open(args.output_path, "w", encoding="utf-8") as fout:
        for ref in pred_ref:
            fout.write(ref + "\n")

    with open(args.output_path + ".reference.txt", "w", encoding="utf-8") as fout:
        targets = [example["target"] for example in data_loader.dataset]
        for target in targets:
            fout.write(target + "\n")

    print("\nSave inference result into: %s" % args.output_path)

    if "target" in data_loader.dataset[0].keys():
        targets = [example["target"] for example in data_loader.dataset]
        bleu4_score = calc_bleu(pred_ref, targets)

    model.train()
    return bleu4_score


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    run(args)
