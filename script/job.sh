#!/bin/bash
#SBATCH --job-name=qwen_baseline         # Job name
#SBATCH --time=05:00:00              # Time limit hh:mm:ss
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                    # Memory per node
#SBATCH --gres=gpu:3                 # Request 1 GPU
#SBATCH --mail-type=END,FAIL         # Notifications for job done & fail

# python -u  prompting_extract_LLMs_aspect.py --input_path ../data/Amasum/min_10_revs_filt_complete/home_kitchen --config config/aspect_extract.yaml --task aspect --output_path ../data/Amasum/min_10_revs_filt_complete/home_kitchen/test_result >& ../log/qwen_home_kitchen_scheme2.log 
python -u  baseline.py --input_path ../data/Amasum/min_10_revs_filt_complete/shoes --config config/aspect_extract.yaml --context_length 50000 --task aspect --output_path ../data/Amasum/min_10_revs_filt_complete/shoes/test_result >& ../log/qwen_shoes_baseline.log 
