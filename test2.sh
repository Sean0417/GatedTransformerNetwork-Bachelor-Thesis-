source venv/bin/activate
path='dataset/WalkvsRun.mat'
plot_folder_dir="./pic" 
model_folder_dir="./saved_models"
POCH=60
BATCH_SIZE=3

LR=1e-4
patience=7
train_percentage=0.75
validate_percentage=0
d_model=512
d_hidden=1024
q=8
v=8
head=8
N=8
dropout=0.2
sliding_window_length=21
optimizer_name='Adagrad'
num_exps=50
is_train=true
given_best_model_path="/homes/soxuxiee/GatedTransformerNetwork-Bachelor-Thesis--1/saved_models/2024-01-27-12-49-42_checkpoint.pth"

if [ "$is_train" = true ]; then
    echo 'training, validation and test'
    python main.py --path=$path \
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
else
    echo 'only conduct testing'
    python main.py --path=$path \
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
    --given_best_model_path=$given_best_model_path
fi
deactivate