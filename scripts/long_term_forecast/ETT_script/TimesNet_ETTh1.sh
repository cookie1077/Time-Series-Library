#export CUDA_VISIBLE_DEVICES=2

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/snu/ \
  --data_path '' \
  --model_id SNU_10_10 \
  --model $model_name \
  --data SNU \
  --features MS \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --embed 'fixed' \
  --factor 3 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 

