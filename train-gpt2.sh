export TRAIN_FILE=/root/transformers/wikitext-2-raw/wiki.train.raw
export TEST_FILE=/root/transformers/wikitext-2-raw/wiki.test.raw
python -m torch.distributed.launch --nproc_per_node=8 ./examples/language-modeling/run_language_modeling.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --logging_steps=100 \
    --num_train_epochs=2 \
    --per_device_train_batch_size 4 \
    --overwrite_output_dir

