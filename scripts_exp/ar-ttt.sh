#!/usr/bin/env bash
set -e
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=3
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export CUDA_LAUNCH_BLOCKING=1

OVERWRITE_RUNS=1
TBS=64
NP=1

MODEL_KIND=rmt4

if [ $MODEL_KIND = "rmt4" ]; then
    MEMORY_CELL=modeling_rmt.rmt4:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.rmt4:RecurrentWrapper
elif [ $MODEL_KIND = "metaogd" ]; then
    MEMORY_CELL=modeling_rmt.metaogd:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.metaogd:RecurrentWrapper
elif [ $MODEL_KIND = "rmt4r" ]; then
    MEMORY_CELL=modeling_rmt.rmt4r:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.rmt4r:RecurrentWrapper
elif [ $MODEL_KIND = "masked_retrieval" ]; then
    MEMORY_CELL=modeling_rmt.masked_retrieval:MemoryCell
    RECURRENT_WRAPPER=modeling_rmt.masked_retrieval:RecurrentWrapper
else
    echo Model $MODEL_KIND not found, aborting
    exit 1
fi

MODEL_NAME=llama
MODEL_CFG=unsloth/Llama-3.2-1B
L=4
H=4
D=128

CUR_RUN_NAME=muon_mem_attn
k=2
# N_SEEDS=3
# for (( k=0; k<$N_SEEDS; k++ )); do

NUMS_PAIRSS=(4)
KEY_SIZES=(2)
VALUE_SIZES=(2)
BSS=(64)
ITERSS=(100000)

for (( j=0; j<${#NUMS_PAIRSS[@]}; j++ )); do

VOCAB_SIZE=62
NUM_PAIRS=${NUMS_PAIRSS[j]}
KEY_SIZE=${KEY_SIZES[j]}
VALUE_SIZE=${VALUE_SIZES[j]}

PAIRS_PER_SEGMENT=2
MAX_N_SEGMENTS=$((NUM_PAIRS / PAIRS_PER_SEGMENT + 1))
BS=${BSS[j]}
ITERS=${ITERSS[j]}
SEGMENT_SIZE=$((PAIRS_PER_SEGMENT * (KEY_SIZE + VALUE_SIZE + 2)))

LR=3e-04

N_MEM_TOKENS=4
INNER_LR=2e-02
INNER_STEPS=3
INNER_CLIP_VALUE=None
INNER_CLIP_NORM=None
INNER_OPTIM=muon
INNER_MOMENTUM_MODE=None
USE_WRITE_HEAD=true
USE_MEM_ATTN=true

RUN_NAME=${MODEL_NAME}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}
RUN_NAME=${RUN_NAME}_K${INNER_STEPS}_ilr${INNER_LR}_${INNER_OPTIM}
if [ "$INNER_CLIP_VALUE" != "None" ]; then
  RUN_NAME=${RUN_NAME}_icv${INNER_CLIP_VALUE}
fi
if [ "$INNER_CLIP_NORM" != "None" ]; then
  RUN_NAME=${RUN_NAME}_icn${INNER_CLIP_NORM}
fi
if [ "$USE_WRITE_HEAD" = true ]; then
  RUN_NAME=${RUN_NAME}_whead
fi
RUN_NAME=${RUN_NAME}_bs${TBS}_lr${LR}

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
ACCEL_CONFIG=./accel_configs/exp/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
cd accel_configs/
python create_config.py \
        --train_batch_size $TBS \
        --train_micro_batch_size_per_gpu $BS \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --np $NP \
        --gradient_clipping 1.0 \
        --prefix deepspeed
cd ..

EXP_PATH="./runs/ar_ttt/K${KEY_SIZE}V${VALUE_SIZE}P${NUM_PAIRS}S${MAX_N_SEGMENTS}V${VOCAB_SIZE}/${RUN_NAME}/run_${CUR_RUN_NAME}_${k}"

if [[ ($OVERWRITE_RUNS -eq 1 || ! -d $EXP_PATH) && $NUM_PAIRS -ne 0 ]]; then

accelerate launch \
    --main_process_port $((29020+$k)) \
    --config_file $ACCEL_CONFIG \
    \
    run_finetuning_associative_retrieval.py \
    --model_path $EXP_PATH \
    --model_cfg $MODEL_CFG \
    --model_hidden $D --model_layers $L --model_heads $H \
    --model_cls transformers:AutoModelForCausalLM \
    --memory_cell_cls $MEMORY_CELL --recurrent_wrapper_cls $RECURRENT_WRAPPER \
    --segment_size $SEGMENT_SIZE \
    --num_mem_tokens $N_MEM_TOKENS \
    --max_n_segments $MAX_N_SEGMENTS \
    --vocab_size $((VOCAB_SIZE-3)) --key_size $KEY_SIZE --value_size $VALUE_SIZE --num_pairs $NUM_PAIRS \
    --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
    --iters $ITERS \
    --num_training_steps $((ITERS*2)) \
    --reset_optimizer --reset_lr --reset_iteration \
    --optimizer AdamW --weight_decay 0.001 \
    --lr ${LR} --lr_scheduler linear --num_warmup_steps 10000 \
    --log_interval 200 --valid_interval 1000 \
    --optimize_metric exact_match --optimize_mode max --best_metric_value 1.0 \
    --show_valid_examples 5 \
    --seed $(($k+42)) \
    --dataset_path ./datasets/associative_retrieval \
    --layers_attr model.layers \
    --train_size 1000000 --valid_size 10000 --test_size 10000 \
    $( [ "$INNER_CLIP_VALUE" != "None" ] && echo "--inner_clip_value $INNER_CLIP_VALUE" ) \
    $( [ "$INNER_CLIP_NORM" != "None" ] && echo "--inner_clip_norm $INNER_CLIP_NORM" ) \
    $( [ "$INNER_MOMENTUM_MODE" != "None" ] && echo "--momentum_mode $INNER_MOMENTUM_MODE" ) \
    $( [ "$USE_WRITE_HEAD" = true ] && echo "--use_write_head" ) \
    $( [ "$USE_MEM_ATTN" = true ] && echo "--use_mem_attn" ) \
    --init_inner_lr $INNER_LR --inner_steps $INNER_STEPS --inner_optim $INNER_OPTIM \
    --save_best
        

else
echo run $EXP_PATH exists already, with OVERWRITE set to 1
fi

# done
done
echo "done"