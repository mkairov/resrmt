#!/usr/bin/env bash
set -e
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export CUDA_LAUNCH_BLOCKING=1
NP=2

MODEL_TYPE=decoder
BACKBONE_CLS=transformers:AutoModelForCausalLM
NOISE_DATASET=pg19
METRIC=exact_match
OVERWRITE_RUNS=1

for MODEL_KIND in prmt; do

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
elif [ $MODEL_KIND = "prmt" ]; then
    MEMORY_CELL=modeling_rmt.prmt:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.prmt:RecurrentWrapper
else
    echo Model $MODEL_KIND not found, aborting
    exit 1
fi

MODEL_NAME=gpt2  # backbone model
    
ITERS=10000

# for TASK_DATASET in qa1_single-supporting-fact; do
# for TASK_DATASET in qa2_two-supporting-facts; do
for TASK_DATASET in qa3_three-supporting-facts; do
# for TASK_DATASET in qa4_two-arg-relations; do

# for TASK_DATASET in qa1_single-supporting-fact qa3_three-supporting-facts qa4_two-arg-relations; do

for LR in 5e-05; do

TBS=64
for SEGMENT_SIZE in 128; do
MAX_N_SEGMENTSS=(0 0 3 0 0 4)
BSS=(0 0 8 0 0 4)

for (( j=2; j<${#MAX_N_SEGMENTSS[@]}; j++ )); do

MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BS=${BSS[j]}

j1=$((j-1))
SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j1]}

j2=$((j-2))
SRC_SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j2]}

if [ $MAX_N_SEGMENTS -ne 0 ]; then

for MEMORY_SIZE in 16; do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))

SCHEDULER=linear
OPTIMIZER=AdamW
WEIGHT_DECAY=1e-02

for RES_MEM_COUNT in 0; do

for N in qa1_babi_cur1 qa1_babi_cur2 qa1_babi_cur3; do

K2=-1 # BPTT unroll length

NP=$NP  
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

MODEL_PATH="/home/mkairov/rmt/runs/${TASK_DATASET}/${MODEL_NAME}/${MODEL_KIND}/lr${LR}_${SCHEDULER}_${OPTIMIZER}_wd${WEIGHT_DECAY}_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_N_SEGMENTS}-${MAX_N_SEGMENTS}/run_${N}"

if [[ $OVERWRITE_RUNS -eq 1 || ! -f "${MODEL_PATH}/metrics.json" ]]; then
# if [ ! -d $MODEL_PATH -o $OVERWRITE_RUNS -eq 1 ]; then

echo RUNNING: MODEL_KIND $MODEL_KIND TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE RES_MEM_COUNT $RES_MEM_COUNT SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

MODEL_CPT="/home/mkairov/rmt/runs/${TASK_DATASET}/${MODEL_NAME}/${MODEL_KIND}/lr${LR}_${SCHEDULER}_${OPTIMIZER}_wd${WEIGHT_DECAY}_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_resmem${RES_MEM_COUNT}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_SRC_N_SEGMENTS}-${SRC_N_SEGMENTS}/run_${N}/model_best"

if [ ! -d $MODEL_CPT ]; then
    echo checkpoint not found, training from scratch
    MODEL_CPT=""
else
    echo checkpoint found
    MODEL_CPT="--model_cpt ${MODEL_CPT}"
fi

accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29003 run_finetuning_babilong_resrmt.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /home/mkairov/rmt/data/tasks_1-20_v1-2/en-10k \
        --model_path $MODEL_PATH $MODEL_CPT \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --res_mem_count $RES_MEM_COUNT \
        --tokenizer gpt2 \
        --vary_n_segments \
        --batch_size $BS  \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --num_training_steps $((ITERS*2)) \
        --iters $ITERS \
        --reset_optimizer --reset_lr --reset_iteration \
        --save_best \
        --k2 $K2 \
        --optimizer $OPTIMIZER --weight_decay $WEIGHT_DECAY \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS / 10)) \
        --data_n_workers 2 \
        --log_interval 25 --valid_interval 100 \
        --optimize_metric $METRIC --optimize_mode max --best_metric_value 1.0 \
        --show_valid_examples 5 \
        --seed $(($N+42)) \
        --layers_attr transformer.h \
        --d_mem 64 \
        --clip_grad_norm 1.0 --max_n_facts 15 --early_stopping_patience 25 --aggr_type full
else

echo run $MODEL_PATH exists already, delete the previous run or ser OVERWRITE_RUNS to 0

fi

done
done
done

fi

done
done
done
done
done
# done

echo done
