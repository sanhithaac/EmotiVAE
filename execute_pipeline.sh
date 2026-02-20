#!/bin/sh
# execute_pipeline.sh
# -------------------
# End-to-end pipeline: train the EmotiVAE model, generate
# visualisations, sample synthetic faces, and explore the latent space.
#
# Designed for an HPC cluster (LSF) but works locally too — just
# comment out the #BSUB lines.
# ----------------------------------------------------------------

### LSF options (uncomment on cluster) ###
# #BSUB -q gpuv100
# #BSUB -J EmotiVAE_RGB
# #BSUB -n 1
# #BSUB -gpu "num=1:mode=exclusive_process"
# #BSUB -W 20:00
# #BSUB -R "rusage[mem=32GB]"
# #BSUB -B
# #BSUB -N
# mkdir -p logs
# #BSUB -o logs/gpu_%J.out
# #BSUB -e logs/gpu_%J.err

# nvidia-smi
# module load cuda/10.2
# module load python3/3.7.7

pip3 install -r requirements.txt --user

# ---- hyper-parameters ----
total_epochs=2000
lr=0.00025
kld_weight=0.5
batch_count=8
latent_dim=20
img_size=50

# 1. Train
python3 train_model.py \
    --total_epochs=$total_epochs \
    --lr=$lr \
    --kld_weight=$kld_weight \
    --batch_count=$batch_count \
    --latent_dim=$latent_dim \
    --img_size=$img_size

# 2. Training analysis
python3 visualize_training.py --img_size=$img_size --latent_dim=$latent_dim

# 3. Modify expression for a range of levels
for level in -3.0 -2.0 -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 \
             -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 \
              0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9 \
              1.0  1.1  1.2  1.3  1.4  1.5  2.0  3.0
do
    python3 modify_expression.py \
        --expression_level=$level --img_size=$img_size --latent_dim=$latent_dim
done

# 4. Sample synthetic faces from prior
for level in -3.0 -2.0 -1.5 -1.4 -1.3 -1.2 -1.1 -1.0 -0.9 \
             -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.0 \
              0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9 \
              1.0  1.1  1.2  1.3  1.4  1.5  2.0  3.0
do
    python3 generate_from_prior.py \
        --expression_level=$level --img_size=$img_size --latent_dim=$latent_dim
done

# 5. Explore every latent axis
for (( dim=0; dim <= $latent_dim; ++dim ))
do
    python3 explore_latent_axis.py \
        --axis=$dim --img_size=$img_size --latent_dim=$latent_dim
done
