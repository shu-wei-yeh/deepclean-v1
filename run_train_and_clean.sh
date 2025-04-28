#!/bin/bash

########################## ONLY CHANGE INFO here ###################################
FOLDER_TIME=1427978587
TRAIN_TIME=1427978587
TRAIN_DURATION=4096
CLEAN_TIME=$((TRAIN_TIME + TRAIN_DURATION)) 
CLEAN_DURATION=4096

LOW_FREQ_COH_MON=330
HIGH_FREQ_COH_MON=340

LOW_FREQ_BP=330
HIGH_FREQ_BP=340

TARGET=acoustic
####################################################################################

dc-prod-train  \
    --save-dataset True \
    --load-dataset False \
    --fs 4096 \
    --chanslist /home/shuwei.yeh/offline-DeepClean/channel_list/cohmon-K1/${FOLDER_TIME}/${LOW_FREQ_COH_MON}Hz-${HIGH_FREQ_COH_MON}Hz.ini \
    --train-kernel 8 \
    --train-stride 0.25 \
    --pad-mode median \
    --filt-fl ${LOW_FREQ_BP} \
    --filt-fh ${HIGH_FREQ_BP} \
    --filt-order 8 \
    --device cuda \
    --train-frac 0.9 \
    --batch-size 32 \
    --max-epochs 20 \
    --num-workers 4 \
    --lr 3.2e-2 \
    --weight-decay 1e-5 \
    --fftlength 2 \
    --psd-weight 1.0 \
    --mse-weight 0.0 \
    --train-dir ../output_bank/${TARGET}/${LOW_FREQ_BP}Hz_${HIGH_FREQ_BP}Hz/train_out/train_dir-${TRAIN_TIME}-${TRAIN_DURATION} \
    --train-t0 ${TRAIN_TIME} \
    --train-duration ${TRAIN_DURATION}

dc-prod-clean \
    --save-dataset True \
    --fs 4096 \
    --out-dir  ../output_bank/${TARGET}/${LOW_FREQ_BP}Hz_${HIGH_FREQ_BP}Hz/dc_out/outdir-${CLEAN_TIME}-${CLEAN_DURATION} \
    --out-channel K1:CAL-CS_PROC_DARM_STRAIN_DBL_DQ \
    --chanslist /home/shuwei.yeh/offline-DeepClean/channel_list/cohmon-K1/${FOLDER_TIME}/${LOW_FREQ_COH_MON}Hz-${HIGH_FREQ_COH_MON}Hz.ini \
    --clean-kernel 8 \
    --clean-stride 4 \
    --pad-mode median \
    --window hanning \
    --device cuda \
    --train-dir ../output_bank/${TARGET}/${LOW_FREQ_BP}Hz_${HIGH_FREQ_BP}Hz/train_out/train_dir-${TRAIN_TIME}-${TRAIN_DURATION} \
    --out-file K-K1_HOFT_DC-${CLEAN_TIME}-${CLEAN_DURATION}.gwf \
    --clean-t0 ${CLEAN_TIME} \
    --clean-duration ${CLEAN_DURATION}
