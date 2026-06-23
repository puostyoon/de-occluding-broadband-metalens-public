export CUDA_VISIBLE_DEVICES=6
NPROC_PER_NODE=1

TRAINING_DATA_DIR=./dataset/de_occluding_broadband_metalens_training_data
TEST_DATA_DIR=./dataset/de_occluding_broadband_metalens_test_data

NETWORK_DIR=None # if exist, load pretrained network weights from this path
BATCH_SIZE=2 # For patch training, use 14. For full image training, use 1. When this number is 1, the CROP_PATCH_SIZE must be equal to FULL_RES.
CROP_PATCH_SIZE=256 # Should be divisible by 4.
FULL_RES=848  # Should be divisible by 4.
RESULT_PATH=de_occluding_broadband_metalens_neural_network_output

L1_LOSS_WEIGHT=1.0
DA_LOSS_WEIGHT=0.0
SSIM_LOSS_WEIGHT=0.1
PERC_LOSS_WEIGHT=0.1
FDL_LOSS_WEIGHT=0.0001
FDL_LOSS_PHASE_WEIGHT=1.0 # DEFAULT 1.0, SOMETIMES USE 0.01
COBI_LOSS_WEIGHT=0.01
COBI_LOSS_WEIGHT_SP=0.1 # Less value for more spatially aligned dataset
COBI_LOSS_BAND_WIDTH=0.2 # high value for flexible matching
NETWORK_LR=5e-6
WARMUP_EPOCHS=10
T_MAX=8000

LOG_FREQ=10 #10 #10
SAVE_FREQ=50 #50 #200

# store true options:
   # use_perc_loss
   # use_da_loss
   # use_ssim_loss
   # use_fdl_loss
   # use_cobi_loss
   # use_warm_up_stage
   # verbose

########################################################################
# When use positional encoded version,
# Make sure to use pretrained network in the later
########################################################################

torchrun --nproc_per_node $NPROC_PER_NODE --master-port=12121 train_neural_network_ddp.py \
    --use_perc_loss \
    --use_ssim_loss \
    --use_warm_up_stage \
    --log_freq $LOG_FREQ \
    --save_freq $SAVE_FREQ \
    --network_lr $NETWORK_LR \
    --l1_loss_weight $L1_LOSS_WEIGHT \
    --da_loss_weight $DA_LOSS_WEIGHT \
    --ssim_loss_weight $SSIM_LOSS_WEIGHT \
    --perceptual_loss_weight $PERC_LOSS_WEIGHT \
    --fdl_loss_weight $FDL_LOSS_WEIGHT \
    --fdl_loss_phase_weight $FDL_LOSS_PHASE_WEIGHT \
    --cobi_loss_weight $COBI_LOSS_WEIGHT \
    --cobi_loss_weight_sp $COBI_LOSS_WEIGHT_SP \
    --cobi_loss_weight_band_width $COBI_LOSS_BAND_WIDTH \
    --training_data_directory $TRAINING_DATA_DIR \
    --test_data_directory $TEST_DATA_DIR \
    --network_dir $NETWORK_DIR \
    --batch_size $BATCH_SIZE \
    --crop_patch_size $CROP_PATCH_SIZE \
    --full_res $FULL_RES \
    --result_path $RESULT_PATH \
    --T_max $T_MAX \
    --warmup_epochs $WARMUP_EPOCHS \
