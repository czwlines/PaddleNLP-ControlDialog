export CUDA_VISIBLE_DEVICES=1
python -u predict.py \
    --predict_file=./data/test.jsonl \
    --vocab_file=./persona/vocab.txt \
    --model_name_or_path=./save/persona/checkpoints/model_best_ppl \
    --output_path=./predict.txt \
    --logging_steps=10 \
    --batch_size=64 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_predict \
    --max_dec_len=30 \
    --min_dec_len=3 \
    --device=gpu
