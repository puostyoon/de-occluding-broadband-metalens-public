RESULT_PATH=./asset/ckpt/broadband
PARAM=./asset/config/param_camPitch1.85_RGB_pitch395nm_864_semiBroad.py

DEVICE=cuda:1
BRIGHTNESS_CLAMP=1
BRIGHTNESS_REGULARIZER_COEFF=1.0
SHARPNESS_REGULARIZER_COEFF=1
CONTRAST_CLAMP=1
PSF_LOSS_WEIGHT=1.0
MASKED_LOSS_WEIGHT=1
L1_LOSS_WEIGHT=1
DA_LOSS_WEIGHT=1
SSIM_LOSS_WEIGHT=1
PERCEPTUAL_LOSS_WEIGHT=0.0
PHASE_NOISE_STDDEV=0.0 
IMAGE_NOISE_STDDEV=0.0 
POISSON_NOISE_MEAN=0.0 
FENCE_DATASET_DIR=None
DIRT_RAINDROP_DATASET_TRAIN_DIR=./dataset/DIV2K/train
DIRT_RAINDROP_DATASET_VAL_DIR=./dataset/LIU4K_v2_validation_arbitrary
OBSTRUCTION=dirt
SAVE_FREQ=800
LOG_FREQ=100
N_EPOCHS=500
T_MAX=3000
PROPAGATOR=SBL_ASM
RESIZING_METHOD=area
PHASE_INIT=random #Fresnel or random

# action:"store true" options:
# --constant_wvl_phase
# --use_lens
# --use_perc_loss
# --use_da_loss
# --use_ssim_loss
# --use_psf_near_guide_loss
# --train_RGB
# --train_broadband
# --concat
# --far_disparity
# --split_spectrum

python train_learned_split_spectrum_metalens.py \
--constant_wvl_phase \
--train_broadband \
--use_ssim_loss \
--use_da_loss \
--T_max $T_MAX \
--phase_init $PHASE_INIT \
--l1_loss_weight $L1_LOSS_WEIGHT \
--da_loss_weight $DA_LOSS_WEIGHT \
--ssim_loss_weight $SSIM_LOSS_WEIGHT \
--perceptual_loss_weight $PERCEPTUAL_LOSS_WEIGHT \
--masked_loss_weight $MASKED_LOSS_WEIGHT \
--psf_loss_weight $PSF_LOSS_WEIGHT \
--brightness_regularizer_coeff $BRIGHTNESS_REGULARIZER_COEFF \
--sharpness_regularizer_coeff $SHARPNESS_REGULARIZER_COEFF \
--log_freq $LOG_FREQ \
--save_freq $SAVE_FREQ \
--n_epochs $N_EPOCHS \
--resizing_method $RESIZING_METHOD \
--result_path $RESULT_PATH \
--param_file $PARAM \
--device $DEVICE \
--dirt_raindrop_dataset_train_dir $DIRT_RAINDROP_DATASET_TRAIN_DIR \
--dirt_raindrop_dataset_val_dir $DIRT_RAINDROP_DATASET_VAL_DIR \
--obstruction $OBSTRUCTION \
--propagator $PROPAGATOR