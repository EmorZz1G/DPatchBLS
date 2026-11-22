cd ../

python main.py --device cuda --data_path '/tsad_data' --data_name MSL --seed 42 --kernel_dim 2 --sigma 0.6 --n_feature_dim 1 --n_feature_group 4 --n_enhancement_dim 7 --n_enhancement_group 3 \
--n_feature_cascade_layer 4 --n_enhancement_cascade_layer 2 --enhancement_s 0.6 --output_reg 0.125 --use_positional false --max_len 50000 --patches_list '[22, 31, 76, 84]'
