#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 NP=1 ./finetune_babilong_baseline.sh
set -e
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export CUDA_LAUNCH_BLOCKING=1
NP=1
OVERWRITE_RUNS=1

MODEL_TYPE=decoder
# BACKBONE_CLS=base_models.modeling_gpt2:GPT2LMHeadModel
BACKBONE_CLS=base_models.modeling_gpt_neox:GPTNeoXForCausalLM
TASK_NAME=associative_retrieval
METRIC=exact_match

for MODEL_KIND in resrmt; do
# for MODEL_KIND in rmt-br rmt-ms; do

if [ $MODEL_KIND = "rmt" ]; then
    MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
elif [ $MODEL_KIND = "resrmt" ]; then
    MEMORY_CELL=modeling_rmt.resrmt:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.resrmt:RecurrentWrapper
# elif [ $MODEL_KIND = "bwrmt" ]; then
#     BACKBONE_CLS=modeling_rmt.block_resrmt:GPT2ModelWithBlockWiseMemory
#     MEMORY_CELL="none --no_memory_cell"
#     RECURRENT_WRAPPER=modeling_rmt.block_resrmt:RecurrentWrapper
elif [ $MODEL_KIND = "rmt-br" ]; then
    MEMORY_CELL=modeling_rmt.rmt_br:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.rmt_br:RecurrentWrapper
elif [ $MODEL_KIND = "rmt-ms" ]; then
    MEMORY_CELL=modeling_rmt.rmt_ms:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.rmt_ms:RecurrentWrapper
elif [ $MODEL_KIND = "armt" ]; then
    MEMORY_CELL=modeling_rmt.armt:AssociativeMemoryCell
    RECURRENT_WRAPPER=modeling_rmt.armt:AssociativeRecurrentWrapper
else
    echo Model $MODEL_KIND not found, aborting
    exit 1
fi

for RES_MEM_COUNT in 0; do
for REWRITE in 0; do

MODEL_NAME=gpt-neox
# MODEL_NAME=gpt2
for MEMORY_SIZE in 4; do

TBS=128
INPUT_SIZE=2048

NUMS_PAIRSS=(5 0 10)
KEY_SIZES=(2 2 2)
VALUE_SIZES=(1 1 1)
BSS=(128 128 128)
ITERSS=(10000 10000 10000)

DIM=128
NUM_LAYERS=4

for N in ar_final1 ar_final2 ar_final3; do

for (( j=0; j<${#NUMS_PAIRSS[@]}; j++ )); do

NUM_PAIRS=${NUMS_PAIRSS[j]}
KEY_SIZE=${KEY_SIZES[j]}
VALUE_SIZE=${VALUE_SIZES[j]}
MAX_N_SEGMENTS=$((NUM_PAIRS + 1))
BS=${BSS[j]}
ITERS=${ITERSS[j]}


BLOCK_SIZE=$((KEY_SIZE + VALUE_SIZE + 2))
cd base_models/gptconfigs
python create_config.py --hidden_size $DIM --num_hidden_layers $NUM_LAYERS --num_attention_heads $NUM_LAYERS
cd ../..
# MODEL_CFG=/data/home/admin/rmt/base_models/gptconfigs/gpt2_tiny_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}.json
MODEL_CFG=/data/home/admin/rmt/base_models/gptconfigs/neox_tiny_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}.json

for LR in 3e-04; do

K2=${MAX_N_SEGMENTS}

for SEGMENT_ORDERING in regular; do

for SCHEDULER in linear; do

if [ $REWRITE -eq 1 ]; then
    echo retrieval with key overwriting
    TASK_TYPE=rewrite
    REWRITE_FLAG="--rewrite_setting"
else
    echo retrieval with unique pairs
    TASK_TYPE=remember
    REWRITE_FLAG=""
fi

if [[ $j -gt 0 ]]; then
    PREV_NUM_PAIRS=${NUMS_PAIRSS[j-1]}
    PREV_MAX_N_SEGMENTS=$((PREV_NUM_PAIRS + 1))
    if [[ $PREV_NUM_PAIRS -ne 0 ]]; then
        MODEL_CPT=../runs/${TASK_NAME}/${TASK_TYPE}/${MODEL_NAME}/${MODEL_KIND}/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZES[j-1]}-v${VALUE_SIZES[j-1]}-p${PREV_NUM_PAIRS}-${PREV_MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${PREV_MAX_N_SEGMENTS}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N 
    else
        MODEL_CPT=None
    fi
else
    MODEL_CPT=None
fi

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
ACCEL_CONFIG=/data/home/admin/rmt/accel_configs/exp/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
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

MODEL_PATH="/data/home/admin/rmt/runs/${TASK_NAME}/${TASK_TYPE}/${MODEL_NAME}/${MODEL_KIND}/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZE}-v${VALUE_SIZE}-p${NUM_PAIRS}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N"

if [[ ($OVERWRITE_RUNS -eq 1 || ! -d $MODEL_PATH) && $NUM_PAIRS -ne 0 ]]; then
# if [ ($OVERWRITE_RUNS -eq 1 -o ! -d $MODEL_PATH) -a $NUM_PAIRS -ne 0 ]; then

echo gradient accumulation steps $GRAD_ACC_STEPS

echo RUNNING: TASK_NAME TASK_TYPE MEMORY_SIZE KEY_SIZE VALUE_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $TASK_TYPE $MEMORY_SIZE $KEY_SIZE $VALUE_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS $LR $N
accelerate launch --config_file $ACCEL_CONFIG --main_process_port $((28000+$MODEL_KIND+$NUM_PAIRS+$N)) run_finetuning_associative_retrieval.py \
        --task_name $TASK_NAME \
        --model_path $MODEL_PATH \
        --model_cfg $MODEL_CFG \
        --model_cls $BACKBONE_CLS \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --segment_size $BLOCK_SIZE \
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
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval 100 --valid_interval 500 \
        --optimize_metric $METRIC --optimize_mode max --best_metric_value 1.0 \
        --show_valid_examples 5 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0 \
        --dataset_path /data/home/admin/rmt/datasets/associative_retrieval \
        --layers_attr gpt_neox.layers \
        --train_size 100000 \
        --valid_size 1000 \
        --test_size 10000 \
        --aggr_type full \
        --vary_n_segments \
        --res_mem_count $RES_MEM_COUNT $REWRITE_FLAG \
        --reset_optimizer --reset_lr \
        --save_best
        
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
done
done
echo "done"