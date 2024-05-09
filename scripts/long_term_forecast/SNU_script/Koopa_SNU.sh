model_name=Koopa

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/snu/ \
  --data_path '' \
  --model_id SNU_30_30 \
  --model $model_name \
  --select_features '0 1 2 4 5 62 63 64' \
  --data SNU \
  --features MS \
  --seq_len 30 \
  --pred_len 15 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 1 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --itr 1




