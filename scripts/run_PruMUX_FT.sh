#!/bin/bash
#SBATCH --job-name=prumux        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G        # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=23:55:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2021.5
conda activate torch-env


glue_low=(MRPC RTE STSB CoLA)
glue_high=(MNLI QQP QNLI SST2)

proj_dir=.

code_dir=${proj_dir}

TASK=$1
SPARSITY=$2
SUFFIX=sparsity$SPARSITY
EX_CATE=PruMUX
PRUNING_TYPE=None
SPARSITY_EPSILON=0.01

# Datamux args
NUM_INSTANCES=${3}

# task and data
task_name=${TASK}
data_dir=$proj_dir/data/glue_data/${task_name}

task_lower=$(echo "$1" | awk '{print tolower($0)}')

# pretrain model
model_name_or_path=princeton-nlp/muxbert_base_${task_lower}_gaussian_hadamard_index_pos_${NUM_INSTANCES}

# logging & saving
logging_steps=100
save_steps=0


# train parameters
max_seq_length=128
bs=32
batch_size=`expr ${NUM_INSTANCES} \* ${bs}`
learning_rate=2e-5
reg_learning_rate=0.01
epochs=20 

# seed
seed=57

# hyperparameters
distill_layer_loss_alpha=$4 
distill_ce_loss_alpha=$5 
# 2: fix hidden layers, 3: min distance matching without restriction, 4: min distance matching with restriction
layer_distill_version=$6 

# output dir
ex_name_suffix=${SUFFIX}
ex_name=${task_name}_${ex_name_suffix}_${NUM_INSTANCES}_${layer_distill_version}_${distill_layer_loss_alpha}
ex_cate=${EX_CATE}
PRUNED_MODEL_PATH=${proj_dir}/out/${task_name}/${ex_cate}/${ex_name}/best

# pruning and distillation
pruning_type=${PRUNING_TYPE}
target_sparsity=${SPARSITY}
distillation_path=/scratch/gpfs/yushans/datamux-prune-pretrain/princeton-nlp/muxbert_base_${task_lower}_gaussian_hadamard_index_pos_${NUM_INSTANCES}
distill_temp=2

scheduler_type=linear


if [[ " ${glue_low[*]} " =~ ${task_name} ]]; then
    eval_steps=50
    epochs=100
    start_saving_best_epochs=50
    prepruning_finetune_epochs=4
    lagrangian_warmup_epochs=20
fi

if [[ " ${glue_high[*]} " =~ ${task_name} ]]; then
    eval_steps=500
    prepruning_finetune_epochs=0
    lagrangian_warmup_epochs=2
fi

pretrained_pruned_model=None

# FT after pruning
if [[ $pruning_type == None ]]; then
  pretrained_pruned_model=${PRUNED_MODEL_PATH}
  learning_rate=$7
  scheduler_type=none
  output_dir=$pretrained_pruned_model/FT-lr${learning_rate}
  epochs=40
  bs=64
  batch_size=`expr ${NUM_INSTANCES} \* ${bs}`
fi

mkdir -p $output_dir

# Datamux args
DEMUXING="index_pos"
MUXING="gaussian_hadamard"
CONFIG_NAME="configs/bert_base.json"
LEARNING_RATE=5e-5
LEARN_MUXING=0
CONTINUE_TRAIN=0
DO_TRAIN=0
DO_EVAL=0

RANDOM_ENCODING_NORM=1
RETRIEVAL_PERCENTAGE=1.0
RETRIEVAL_PRETRAINING=0
RETRIEVAL_LOSS_COEFF=0
TASK_LOSS_COEFF=1.0
SHOULD_MUX=1
DATALOADER_DROP_LAST=1
OUTPUT_DIR_BASE="checkpoints/finetune"

python3 $code_dir/run_glue_prune.py \
    --output_dir ${output_dir} \
    --logging_steps ${logging_steps} \
    --task_name ${task_name} \
    --model_name_or_path ${model_name_or_path} \
    --ex_name ${ex_name} \
    --do_train \
    --do_eval \
    --max_seq_length ${max_seq_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --reg_learning_rate ${reg_learning_rate} \
    --num_train_epochs ${epochs} \
    --overwrite_output_dir \
    --save_steps ${save_steps} \
    --eval_steps ${eval_steps} \
    --evaluation_strategy steps \
    --seed ${seed} \
    --pruning_type ${pruning_type} \
          --report_to none \
    --pretrained_pruned_model ${pretrained_pruned_model} \
    --target_sparsity $target_sparsity \
    --freeze_embeddings \
    --do_distill \
    --do_layer_distill \
    --distillation_path $distillation_path \
    --distill_ce_loss_alpha $distill_ce_loss_alpha \
    --distill_loss_alpha $distill_layer_loss_alpha \
    --distill_temp $distill_temp \
    --scheduler_type $scheduler_type \
    --layer_distill_version $layer_distill_version \
    --prepruning_finetune_epochs $prepruning_finetune_epochs \
    --lagrangian_warmup_epochs $lagrangian_warmup_epochs \
    --tokenizer_name bert-base-uncased \
    --config_name ${CONFIG_NAME} \
    --dataloader_drop_last $DATALOADER_DROP_LAST \
    --retrieval_percentage $RETRIEVAL_PERCENTAGE \
    --retrieval_loss_coeff $RETRIEVAL_LOSS_COEFF \
    --task_loss_coeff $TASK_LOSS_COEFF \
    --retrieval_pretraining ${RETRIEVAL_PRETRAINING} \
    --num_instances ${NUM_INSTANCES} \
    --muxing_variant ${MUXING} \
    --demuxing_variant ${DEMUXING} \
    --should_mux ${SHOULD_MUX} \
    --gaussian_hadamard_norm ${RANDOM_ENCODING_NORM} \
    --num_hidden_demux_layers 3 \
    --gradient_accumulation_steps 4 \
    --learn_muxing ${LEARN_MUXING} 2>&1 | tee ${output_dir}/log.txt
