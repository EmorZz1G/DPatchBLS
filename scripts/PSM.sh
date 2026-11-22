cd ../

python main.py --device cuda --data_path '/tsad_data' --data_name PSM --seed 42 --kernel_dim 2 --sigma 1.9 --n_feature_dim 2 --n_feature_group 3 --n_enhancement_dim 4 --n_enhancement_group 3 \
--n_feature_cascade_layer 1 --n_enhancement_cascade_layer 3 --enhancement_s 0.5 --output_reg 0.5 --use_positional false --max_len 50000 --patches_list '[6, 42, 88, 98]'

