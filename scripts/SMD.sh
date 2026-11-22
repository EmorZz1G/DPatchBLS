cd ../

python /main.py --device cuda --data_path '/tsad_data' --data_name SMD --seed 42 --kernel_dim 5 --sigma 1.4 --n_feature_dim 9 --n_feature_group 5 --n_enhancement_dim 3 --n_enhancement_group 4 \
--n_feature_cascade_layer 4 --n_enhancement_cascade_layer 5 --enhancement_s 0.2 --output_reg 0.5 --use_positional false --max_len 50000 --patches_list '[68, 81, 86, 98]'
