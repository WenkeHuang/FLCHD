communication_epoch=50
parti_num=4
seed=42
device_id=0

local_epoch=5
model="fedavg"
chdtype="multi"

leave_one_out=true  
leave_out_hospital="CSSFY"

use_atest=false 
atest_interval=5


if [ "$chdtype" == "multi" ]; then
    dataset="fl_chd_multi" 
elif [ "$chdtype" == "binary" ]; then
    dataset="fl_chd_new" 
fi

if [ "$leave_one_out" == "true" ]; then
    parti_num=3 
fi


atest_args=""
if [ "$use_atest" == "true" ]; then
    atest_args="--use_atest --atest_interval ${atest_interval}"
    echo "A-Test validation enabled with interval: ${atest_interval}"
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
HF_ENDPOINT=https://hf-mirror.com \
HF_CACHE_DIR=/data0/lyx_mydata/.cache/transformers \
python main.py \
    --communication_epoch ${communication_epoch} \
    --local_epoch ${local_epoch} \
    --parti_num ${parti_num} \
    --seed ${seed} \
    --model ${model} \
    --dataset ${dataset} \
    --local_epoch ${local_epoch} \
    --device_id ${device_id} \
    --chdtype ${chdtype} \
    --leave_one_out ${leave_one_out} \
    --leave_out_hospital ${leave_out_hospital} \
    ${atest_args}