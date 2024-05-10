model_name=Mamba
pred_len=30

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/snu/ \
  --data_path '' \
  --model_id SNU_30_30 \
  --model $model_name \
  --select_features '0 3 4 5' \
  --data SNU \
  --features MS \
  --seq_len $pred_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 4 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/snu/ \
  --data_path '' \
  --model_id SNU_30_30 \
  --model $model_name \
  --select_features '0 1 2 3 4 62 63 64 92 93 94 54 69 70 71 72' \
  --data SNU \
  --features MS \
  --seq_len $pred_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 16 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 1 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \
