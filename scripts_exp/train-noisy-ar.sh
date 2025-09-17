#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 NP=1 ./finetune_babilong_baseline.sh
set -e
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=2
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export CUDA_LAUNCH_BLOCKING=1
NP=1
OVERWRITE_RUNS=1

MODEL_TYPE=decoder
# BACKBONE_CLS=base_models.modeling_gpt2:GPT2LMHeadModel
# BACKBONE_CLS=base_models.modeling_gpt_neox:GPTNeoXForCausalLM
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=noisy_ar
METRIC=loss

for MODEL_KIND in rmt4; do
# for MODEL_KIND in rmt; do
# for MODEL_KIND in rmt-br rmt-ms; do

if [ $MODEL_KIND = "rmt" ]; then
    MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
elif [ $MODEL_KIND = "resrmt" ]; then
    MEMORY_CELL=modeling_rmt.resrmt:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.resrmt:RecurrentWrapper
elif [ $MODEL_KIND = "rmt-br" ]; then
    MEMORY_CELL=modeling_rmt.rmt_br:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.rmt_br:RecurrentWrapper
elif [ $MODEL_KIND = "rmt-ms" ]; then
    MEMORY_CELL=modeling_rmt.rmt_ms:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.rmt_ms:RecurrentWrapper
elif [ $MODEL_KIND = "armt" ]; then
    MEMORY_CELL=modeling_rmt.armt:AssociativeMemoryCell
    RECURRENT_WRAPPER=modeling_rmt.armt:AssociativeRecurrentWrapper
elif [ $MODEL_KIND = "rmt4" ]; then
    MEMORY_CELL=modeling_rmt.rmt4:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.rmt4:RecurrentWrapper
elif [ $MODEL_KIND = "prmt4" ]; then
    MEMORY_CELL=modeling_rmt.prmt4:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.prmt4:RecurrentWrapper
else
    echo Model $MODEL_KIND not found, aborting
    exit 1
fi

for RES_MEM_COUNT in 0; do

# MODEL_NAME=gpt-neox
# MODEL_NAME=gpt2
MODEL_NAME=llama
MODEL_CFG=unsloth/Llama-3.2-1B
for MEMORY_SIZE in 8; do

TBS=64
INPUT_SIZE=64

# DIFFICULT EXPERIMENT

# NUMS_PAIRS=(10)
# KEY_SIZES=(2)
# VALUE_SIZES=(2)
# BSS=(128)
# MAX_N_SEGMENTS=5
# MIN_SEGMENT_SIZE=32
# BLOCK_SIZE=64

# SIMPLE EXPERIMENT

NUMS_PAIRS=(1)
KEY_SIZES=(4)
VALUE_SIZES=(4)
BSS=(64)
MAX_N_SEGMENTSS=(1)
MIN_SEGMENT_SIZES=(16)
BLOCK_SIZES=(32)
INNER_STEPSS=(3)

# 2 SEGMENTS (MEMORY TEST)

# NUMS_PAIRS=(2)
# KEY_SIZES=(4)
# VALUE_SIZES=(4)
# BSS=(64)
# MAX_N_SEGMENTSS=(2)
# MIN_SEGMENT_SIZES=(16)
# BLOCK_SIZES=(32)
# INNER_STEPSS=(2)

# CURRICULUM

# NUMS_PAIRS=(1 2 2 2 2 10)
# KEY_SIZES=(4 4 4 4 4 2)
# VALUE_SIZES=(4 4 4 4 4 2)
# BSS=(64 64 64 64 64 64)
# MAX_N_SEGMENTSS=(1 1 2 4 4 4)
# MIN_SEGMENT_SIZES=(16 16 16 16 32 32)
# BLOCK_SIZES=(32 32 32 32 64 64)
# INNER_STEPSS=(2 2 2 2 3 3)

DIM=128
NUM_LAYERS=4
NUM_HEADS=4

# for N in testttt; do
for N in qqpp; do

for (( j=0; j<${#NUMS_PAIRS[@]}; j++ )); do

NUM_PAIRS=${NUMS_PAIRS[j]}
KEY_SIZE=${KEY_SIZES[j]}
VALUE_SIZE=${VALUE_SIZES[j]}
BS=${BSS[j]}

MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
MIN_SEGMENT_SIZE=${MIN_SEGMENT_SIZES[j]}
BLOCK_SIZE=${BLOCK_SIZES[j]}
INNER_STEPS=${INNER_STEPSS[j]}

ITERS=200000
for LR in 1e-04; do

K2=${MAX_N_SEGMENTS}


# for SCHEDULER in linear; do
for SCHEDULER in constant_with_warmup; do

if [[ j -gt 0 ]]
then
    PREV_NUM_PAIRS=${NUMS_PAIRS[j-1]}
    PREV_MAX_N_SEGMENTS=$((PREV_NUM_PAIRS + 1))
    MODEL_CPT=../runs/${TASK_NAME}/${MODEL_NAME}/${MODEL_KIND}/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZES[j-1]}-v${VALUE_SIZES[j-1]}-p${PREV_NUM_PAIRS}-${PREV_MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_bptt-${K2}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N 
else
    MODEL_CPT=None
fi

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
ACCEL_CONFIG=/home/mkairov/rmt/accel_configs/exp/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
cd accel_configs/
python create_config.py \
        --bf16 \
        --train_batch_size $TBS \
        --train_micro_batch_size_per_gpu $BS \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --np $NP \
        --gradient_clipping 1.0 \
        --prefix deepspeed
cd ..

MODEL_PATH="/home/mkairov/rmt/runs/${TASK_NAME}/${MODEL_NAME}/${MODEL_KIND}/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZE}-v${VALUE_SIZE}-p${NUM_PAIRS}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_bptt-${K2}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N"

if [ $OVERWRITE_RUNS -eq 1 -o ! -d $MODEL_PATH ]; then

echo gradient accumulation steps $GRAD_ACC_STEPS

echo RUNNING: TASK_NAME MEMORY_SIZE KEY_SIZE VALUE_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $KEY_SIZE $VALUE_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS $LR $N
accelerate launch --config_file $ACCEL_CONFIG --main_process_port $((29503+$TBS)) run_finetuning_noisy_ar.py \
        --task_name $TASK_NAME \
        --model_path $MODEL_PATH \
        --model_cfg $MODEL_CFG \
        --model_cls $BACKBONE_CLS \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --segment_size $BLOCK_SIZE \
        --min_segment_size $MIN_SEGMENT_SIZE \
        --key_size $KEY_SIZE \
        --value_size $VALUE_SIZE \
        --num_pairs $NUM_PAIRS \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --num_training_steps $((ITERS*2)) \
        --reset_optimizer --reset_lr --reset_iteration \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.0 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 1000 \
        --data_n_workers 2 \
        --log_interval 100 --valid_interval 500 \
        --optimize_metric $METRIC --optimize_mode min --best_metric_value 0.0 \
        --show_valid_examples 5 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0 \
        --dataset_path /home/mkairov/rmt/datasets/associative_retrieval \
        --layers_attr model.layers \
        --train_size 100000 \
        --valid_size 1000 \
        --test_size 10000 \
        --aggr_type full \
        --init_inner_lr 2.5 --inner_steps $INNER_STEPS --inner_clip_norm 1.0 \
        --res_mem_count $RES_MEM_COUNT \
        --reset_optimizer --reset_lr \
        --save_best \
        --model_cpt $MODEL_CPT
        # --backbone_cpt /home/admin/rmt/runs/neox_noisy_ar.pth
        # --use_generate_on_valid --save_best
        
        # --layers_attr transformer.h \
        # --early_stopping_patience 10 
        # --model_cpt $MODEL_CPT
        # --use_generate_on_valid \

else
echo run $MODEL_PATH exists already, with OVERWRITE set to 1
fi

done
done
done
done
done
done
done
echo "done"