#python compute_sim.py --model_name grit_instruct_512_file --dataset_name AD --head_count 5 --is_file --mode max
#python compute_sim.py --model_name grit_instruct_512_file --dataset_name AD --head_count 2 --is_file --mode mean
#python compute_sim.py --model_name grit_instruct_512_file --dataset_name AD --head_count 1 --is_file --mode mean
#python compute_sim.py --model_name grit_instruct --dataset_name AD --head_count 1 --mode mean
python compute_sim.py --model_name path --dataset_name AD --head_count 1 --mode mean --is_train
