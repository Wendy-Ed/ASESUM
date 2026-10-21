#!/bin/bash
#SBATCH --job-name=gpu_job          # Job name
#SBATCH --time=02:00:00              # Time limit hh:mm:ss
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=16G                    # Memory per node
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mail-type=END,FAIL         # Notifications for job done & fail


python -u evaluation_script.py --gt_dir '../../data/Amasum/min_10_revs_filt_complete/elec/' --res_dir '../../data/Amasum/min_10_revs_filt_complete/elec/test_result/' --metric summac >& ../../log/evaluation_argscheme_elec.log --device cuda
# python -u evaluation_script.py --gt_dir '' --res_p ../../Tom/hercules/output/home_kitchen_train_result_postprocess.json --metric summac >& ../log/evaluation_hercules_home-kitchen_true.log