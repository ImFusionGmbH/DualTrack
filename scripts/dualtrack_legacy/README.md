## DualTrack Legacy Code

The functionality of the scripts in this folder have been replaced by the `.yaml` config system and the `train.py` and `evaluate.py` scripts in the root of the source tree. We keep the legacy versions documented here.

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
This will generate useful visualizations, a table of error values per scan, and the averaged error metrics, along with some error plot visualizations for each scan. We provide the pretrained weights of our best DualTrack model at [MODEL_URL_PLACEHOLDER]. To evaluate this model, simply download it, name it as `$(pwd)/trained_models/dualtrack_final.pt`, and run the following:
```bash 
python scripts/dualtrack/train_fusion_model.py --local_encoder_name dualtrack_loc_enc_stg3_legacy test --model_weights trained_models/dualtrack_final.pt --dataset=tus-rec-val --output experiments/dualtrack/test
```
which will reproduce the results from the bottom row of Table 1 in the paper. 
