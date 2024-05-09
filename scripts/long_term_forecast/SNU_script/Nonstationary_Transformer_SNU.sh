model_name=Nonstationary_Transformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/snu/ \
  --data_path '' \
  --model_id SNU_30_30 \
  --model $model_name \
  --data SNU \
  --features MS \
  --select_features '0 1 2 4 5 62 63 64' \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --d_model 128

