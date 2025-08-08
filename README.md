


# Data Preparation

TODO: Describe the pipeline for data preparation

# Run Model Training and Evaluation

## DualTrack

### Pretraining Local Encoder

The first step of DualTrack is to pretrain the local encoder. This is done in three stages. First, we setup up an output directory for the experiments using the following commands:

```bash
loc_enc_dir=experiments/dualtrack/local_encoder
mkdir -p $loc_enc_dir
```

Pretraining step 1 - we pretrain the 3d CNN backbone on small subsequences of images for 5000 epochs (should take 4-5 days on NVIDIA A40 GPU): 
```bash
python scripts/dualtrack/train_local_encoder.py --log_dir=${loc_enc_dir}/stage1 --epochs=5000 --lr=1e-4 --weight_decay=1e-3 --run_validation_every_n_epochs=100 --batch_size=16 --sequence_length_train=16 --augmentations --model=dualtrack_loc_enc_stg1 
```

Pretrain step 2 - we add a vit stage for frame-wise spatial self-attention on top of the frozen CNN backbone of stage 1 using this command: 
```bash
python scripts/dualtrack/train_local_encoder.py --log_dir=${loc_enc_dir}/stage2 --epochs=500 --lr=1e-3 --weight_decay=0 --run_validation_every_n_epochs=10 --batch_size=1 --augmentations --model=dualtrack_loc_enc_stg2 --backbone_weights=${loc_enc_dir}/stage1/checkpoint/last.pt
```

Pretrain step 3 - here we add temporal attention stage and pretrain it on top of the frozen CNN + vit model of stage 1. To substantially improve training speed, we also can use the extra step of pre-computing and caching the features of the CNN + vit model and training on top of these. Here are the relevant commands:
```bash
# export the features of the backbone 
python scripts/dualtrack/train_local_encoder.py --log_dir=${loc_enc_dir}/stage3 --batch_size=1 --model=dualtrack_loc_enc_stg3 --backbone_weights=${loc_enc_dir}/stage2/checkpoint/last.pt --cached_features_file ${loc_enc_dir}/stage3/cached_intermediates.h5 export_features

# train the model on top of the features
python scripts/dualtrack/train_local_encoder.py --log_dir=${loc_enc_dir}/stage3 --batch_size=1 --model=dualtrack_loc_enc_stg3 --backbone_weights=${loc_enc_dir}/stage2/checkpoint/last.pt --cached_features_file ${loc_enc_dir}/stage3/cached_intermediates.h5 --batch_size=1 --lr=1e-4 --epochs 500 --run_validation_every_n_epochs=10 

# you could also skip the export cached features stage, but it will be far slower.
python scripts/dualtrack/train_local_encoder.py --log_dir=${loc_enc_dir}/stage3 --batch_size=1 --model=dualtrack_loc_enc_stg3 --backbone_weights=${loc_enc_dir}/stage2/checkpoint/last.pt --batch_size=1 --lr=1e-4 --epochs 500 --run_validation_every_n_epochs=10 

```

### Pretraining Global Encoder

The second step of DualTrack is to pretrain the global encoder using sparsely sampled subsequences of the ultrasound frames. The global encoder consists of an image backbone and then a transformer temporal self-attention stage. Here we have several options for the image backbone: CNN, iBOT, MedSAM, and USFM. The code can easily be adapted to using other backbones. Note that some backbones require pretrained weights or add dependencies, which we describe later. #TODO

```bash 

GLOBAL_ENCODER_LOG_DIR=experiments/dualtrack/global_encoder

# CNN backbone
python scripts/dualtrack/train_global_encoder.py --model=global_encoder_cnn --use_amp --in_channels=1 --mean 0.5 --std 0.25 --log_dir=${GLOBAL_ENCODER_LOG_DIR}

# iBOT backbone
IBOT_PRETRAINED_WEIGHTS=/path/to/ibot/weights # specify the location of ibot pretraining
python scripts/dualtrack/train_global_encoder.py --model=global_encoder_ibot --use_amp --in_channels=1 --mean 0.5 --std 0.25 --backbone_weights=$IBOT_PRETRAINED_WEIGHTS --log_dir=${GLOBAL_ENCODER_LOG_DIR}

# MedSAM backbone
python scripts/dualtrack/train_global_encoder.py --model=global_encoder_medsam --use_amp --in_channels=3 --mean 0 0 0 --std 1 1 1 --batch_size=4 --log_dir=${GLOBAL_ENCODER_LOG_DIR}

# USFM backbone 
# (usfm used imagenet stats for normalization)
python scripts/final/train_global_encoder.py --model=global_encoder_usfm --use_amp --in_channels=3 --mean 0.485 0.456 0.406 --std 0.228 0.224 0.225 --batch_size=4 --log_dir=${GLOBAL_ENCODER_LOG_DIR}
```

### Training Fusion Model 

The final step is to combine the global and local encoders using a fusion module. Here is the command to run: 

```bash 
python scripts/dualtrack/train_fusion_model.py --log_dir=experiments/dualtrack/fusion_model --local_encoder_name dualtrack_loc_enc_stg3 --local_encoder_ckpt ${loc_enc_dir}/stage3/checkpoint/best.pt --global_encoder_name global_encoder_cnn --global_encoder_ckpt ${GLOBAL_ENCODER_LOG_DIR}/checkpoint/best.pt --mean 0.5 --std 0.25 --in_channels=1 --loc_encoder_intermediates_cache ${loc_enc_dir}/stage3/cached_intermediates.h5
```

### Evaluation

Scripts will log aggregate metrics information from the training and validation sets throughout training. Once we have our final model, to run a full test routine, we can use the following command: 
```bash 
python scripts/dualtrack/train_fusion_model.py --local_encoder_name dualtrack_loc_enc_stg3_legacy test --model_weights experiments/dualtrack/fusion_model/checkpoint/best.pt --dataset=tus-rec-val --output experiments/dualtrack/test
```
This will generate useful visualizations, a table of error values per scan, and the averaged error metrics, along with some error plot visualizations for each scan. We provide the pretrained weights of our best DualTrack model at [TBD](dead_url). To evaluate this model, simply download it, name it as `$(pwd)/trained_models/dualtrack_final.pt`, and run the following:
```bash 
python scripts/dualtrack/train_fusion_model.py --local_encoder_name dualtrack_loc_enc_stg3_legacy test --model_weights trained_models/dualtrack_final.pt --dataset=tus-rec-val --output experiments/dualtrack/test
```
which will reproduce the results from the bottom row of Table 1 in the paper. 


## Baselines 

We reproduced the following baselines for tracking estimation:

### 2-Frame CNN:

We have the 2-Frame CNN method based on [Prevost et al. 2018](https://pubmed.ncbi.nlm.nih.gov/29936399/). To train it, run: 
```bash
python scripts/baselines/run_baseline_2_frame_cnn_reprod.py --log_dir experiments/baselines/2-frame-cnn --scheduler=none --model efficientnet_b1 --epochs 6700 --batch_size=16 --optimizer=adam --epoch_mode tus_rec --validate_every 100 --dataset tus-rec --val_datasets tus-rec --flip_h_prob=.5 --reverse_sweep_prob=0.5 --skip_frame_prob=0.2 
```
To run a full test loop, run: 
```bash 
python scripts/baselines/run_baseline_2_frame_cnn_reprod.py --dataset tus-rec-val --model efficientnet_b1 --epochs 6700 --train_dir experiments/baselines/2-frame-cnn --test_dataset tus-rec-val
```

### DCLNet

To run the DCLNet method of [Guo et al. 2020](https://arxiv.org/abs/2006.07694), run: 

```bash
# train
python scripts/baselines/train_dcnet.py -c scripts/baselines/dcnet.yaml --log_dir=experiments/baselines/dcnet

# test 
python scripts/baselines/train_dcnet.py test --train_dir=experiments/baselines/dcnet --test_dataset tus-rec-val
```

### MoNet

```bash 
# train 
python scripts/baselines/run_monet_baseline.py --epochs=3000 --log_dir=experiments/baselines/monet

# test
python scripts/baselines/run_monet_baseline.py --batch_size=1 --use_full_scan_for_val --log_dir=experiments/baselines/monet --dataset=tus-rec-val test 
```

### Hybrid Transformer

Hybrid transformer is implemented based on the paper "Spatial Position Estimation Method for 3D Ultrasound Reconstruction Based on Hybrid Transfomers" [Ning et al. 2022](https://ieeexplore.ieee.org/abstract/document/9761499?casa_token=_ZXiTiUncyQAAAAA:Zj5R8rlZW4iGlmaNKS-2n7Eo6qCApXZNh99orsoHr3vhzqTSXcL5pf5Vw6wa3NKpLrgj-iuZ2Hg)

The standard method as described in the paper can be run as follows: 
```bash 
python scripts/baselines/run_ning_et_al_reprod.py --hidden_size=128 --log_dir=experiments/baselines/hybrid_transformer 
```

We found it to improve performance and reduce computational complexity of training to implement a 2-stage training setup for this model, where we pretrain the CNN component, export its features, then train the transformer component on top of these features (unlike the original paper which used end to end training). To reproduce these steps, we run the following 

1. pretrain the cnn 

```bash
python scripts/dualtrack/train_local_encoder.py --log_dir=experiments/baselines/hybrid_transformer/stage1 --epochs=5000 --lr=1e-4 --weight_decay=1e-3 --run_validation_every_n_epochs=100 --batch_size=16 --sequence_length_train=16 --augmentations --model=vidrn18_small_window_trck_reg_causal
```

2. export its features 
```bash 
python scripts/dualtrack/train_local_encoder.py --log_dir=experiments/baselines/hybrid_transformer/stage1 --batch_size=1 --model=vidrn18_small_window_trck_reg_causal --model_kwargs checkpoint=experiments/baselines/hybrid_transformer/stage1/checkpoint/best.pt --cached_features_file experiments/baselines/hybrid_transformer/stage1/features.h5 export_features

python scripts/dualtrack/train_local_encoder.py --log_dir=experiments/baselines/hybrid_transformer/stage1 --batch_size=1 --model=vidrn18_small_window_trck_reg_causal --model_kwargs checkpoint=experiments/baselines/hybrid_transformer/stage1/checkpoint/best.pt --cached_features_file experiments/baselines/hybrid_transformer/stage1/features.h5 --dataset=tus-rec-val export_features
``` 

3. train on top of these features
```bash
python scripts/baselines/run_ning_et_al_reprod.py --features_path=data/pre-computed-features/lively-blaze_causal/feats.h5 --hidden_size=128 --log_dir=experiments/baselines/hybrid_transformer/stage2
```

