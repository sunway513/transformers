export SQUAD_DIR=/data/datasets/squad/v1.1/

export HSA_FORCE_FINE_GRAIN_PCIE=1

python3.6 -m torch.distributed.launch --nproc_per_node=8 examples/question-answering/run_squad.py \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir perf \
  --max_steps 1000 \
  --overwrite_output_dir \
  --fp16
