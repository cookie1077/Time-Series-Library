model_name=TimeMixer

seq_len=30
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path  ./dataset/snu/\
  --data_path '' \
  --model_id SNU_30_30 \
  --model $model_name \
  --data SNU \
  --features MS \
  --select_features '0 1 2 4 5 62 63 64' \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 30 \
  --e_layers $e_layers \
  --enc_in 8 \
  --c_out 1 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size 32 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window



  
