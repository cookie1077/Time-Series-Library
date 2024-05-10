# export CUDA_VISIBLE_DEVICES=2

model_name=FEDformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/snu/ \
  --data_path '' \
  --model_id SNU_30_30 \
  --model $model_name \
  --data SNU \
  --features MS \
  --select_features '0 3 4 5' \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --embed 'fixed' \
  --factor 4 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 

"""
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
  --embed 'fixed' \
  --factor 8 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 

"""

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/snu/ \
  --data_path '' \
  --model_id SNU_30_30 \
  --model $model_name \
  --data SNU \
  --features MS \
  --select_features '0 1 2 3 4 62 63 64 92 93 94 54 69 70 71 72' \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 30 \
  --e_layers 2 \
  --d_layers 1 \
  --embed 'fixed' \
  --factor 16 \
  --enc_in 16 \
  --dec_in 16 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 
