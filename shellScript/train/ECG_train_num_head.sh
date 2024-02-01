source venv/bin/activate
project_name="ECG_train_num_head"
path='/homes/soxuxiee/GatedTransformerNetwork-Bachelor-Thesis--1/dataset/ECG.mat'
plot_folder_dir="./pic" 
model_folder_dir="./saved_models"
EPOCH=5
BATCH_SIZE=3

LR=1e-4
patience=7
train_percentage=0.75
validate_percentage=0
d_model=512
d_hidden=1024
q=8
v=8
head_list=(1 2 4 8)
N=8
dropout=0.2
sliding_window_length=21
optimizer_name='Adagrad'
num_exps=1
is_train=true
given_best_model_path="/homes/soxuxiee/GatedTransformerNetwork-Bachelor-Thesis--1/saved_models/ECG/ECG_d_model512_num_encoder8_num_head8_20240131012611_checkpoint.pth"
for head in "${head_list[@]}"
do
    echo 'training, validation and test'
    python main.py --project_name=$project_name \
    --path=$path \
    --plot_folder_dir=$plot_folder_dir \
    --model_folder_dir=$model_folder_dir \
    --EPOCH=$EPOCH \
    --BATCH_SIZE=$BATCH_SIZE \
    --learning_rate=$LR \
    --patience=$patience \
    --train_percentage=$train_percentage \
    --validate_percentage=$validate_percentage \
    --d_model=$d_model \
    --d_hidden=$d_hidden \
    -q=$q \
    -v=$v \
    -head=$head \
    -N=$N \
    --dropout=$dropout \
    --sliding_window_length=$sliding_window_length \
    --optimizer_name=$optimizer_name \
    --num_exps=$num_exps \
    --is_train
done
deactivate