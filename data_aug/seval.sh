

idx=$SLURM_ARRAY_TASK_ID
total=$SLURM_ARRAY_TASK_COUNT
jname=seval-$idx-of-$total-random

ckpt=${1:-"/PATH_TO_YOUR_CHECKPOINT"}
conv_mode=${2:-"qwen_vl"}
temperature=${3:-"0.0"}
num_beams=${4:-1}

OUTDIR=slurm-logs/$ckpt
#_$wname
> $OUTDIR/$jname.err
> $OUTDIR/$jname.out


srun \
    -e $OUTDIR/$jname.err -o $OUTDIR/$jname.out \
    python qwen3_vl/data_aug/video_eval.py \
        --model-path $ckpt --shard $idx --total $total --conv-mode $conv_mode \
        --temperature $temperature --num-beams $num_beams

'''
# Examples (replace with your checkpoint):
# python qwen3_vl/data_aug/video_eval.py --model-path /PATH_TO_YOUR_CHECKPOINT -c
# python qwen3_vl/data_aug/video_eval.py --model-path /PATH_TO_YOUR_CHECKPOINT --conv-mode qwen_vl
python qwen3_vl/data_aug/video_eval.py --shard 7 --total 10


tmp=0.2
beam=1
# python qwen3_vl/data_aug/video_eval.py --model-path /PATH_TO_YOUR_CHECKPOINT -c --temperature $tmp --num-beams $beam


tmp=0
beam=1
sbatch -A cosmos_misc -p interactive,interactive_singlenode,$SLURM_PARTITION -J fz-13b-video-mme-eval \
# sbatch qwen3_vl/data_aug/seval.sh /PATH_TO_CHECKPOINT qwen_vl $tmp $beam
# python qwen3_vl/data_aug/video_eval.py --model-path /PATH_TO_CHECKPOINT -c --temperature $tmp --num-beams $beam
'''
