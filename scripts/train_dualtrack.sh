sbatch scripts/train.sh python scripts/dualtrack/train_fusion_model.py \
    --model dualtrack_tus_rec_2024 \
    --use_amp \
    --val_every 1 \
