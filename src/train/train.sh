export TRAIN_PATH=../../data/concat_train_random.jsonl
export DEV_PATH=../../data/concat_dev_random.jsonl

python train.py \
  --seed 10 \
  --train_data $TRAIN_PATH \
  --dev_data $DEV_PATH \
  --model_type gpt2 \
  --model_name gpt2 \
  --output_dir ../../output \
  --max_epochs 1 \
  --train_batch_size 16 \
  --dev_batch_size 8 \
  --device cuda:0 \
  --accumulate_grad_batches 2 \
  --logging_steps 100 \
  --eval_steps 1000 \
  --replay_interval 1000 \
  "$@"
