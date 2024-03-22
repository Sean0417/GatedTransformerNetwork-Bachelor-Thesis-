source venv/bin/activate
# project_name="KickvsPunch_train_with_only__(16,1,1)"
# project_name="KickvsPunch_train_with_only_channelwise"
project_name="KickvsPunch_train_Grid_search_last"
path='dataset/KickvsPunch.mat'
plot_folder_dir="./pic" 
model_folder_dir="./saved_models"
EPOCH=100
BATCH_SIZE=3
attn_list=("ProbSparse_attn" "normal_attn")
# attn_list=("longformer_attn")
# attn_list=('normal_attn')
LR=1e-4
patience=7
train_percentage=0.75
validate_percentage=0
d_model_list=(512)
d_hidden=1024
q=8
v=8
# head=12
head=1
head_list=(1)
N_list=(8)
dropout=0.2
sliding_window_length=21
optimizer_name='Adagrad'
num_exps=50
is_train=true
given_best_model_path="/homes/soxuxiee/GatedTransformerNetwork-Bachelor-Thesis--1/saved_models/ECG/ECG_d_model512_num_encoder8_num_head8_20240131012611_checkpoint.pth"
for attn_type in "${attn_list[@]}"
do
    for d_model in "${d_model_list[@]}"
    do
       for N in "${N_list[@]}"
       do
          for head in "${head_list[@]}"
          do
                echo 'training, validation and test'
                echo "Attention Module:$attn_type"
                echo ”d_model:$d_model, head:$head, encoder:$N“
                python main.py --project_name=$project_name \
                    --attn_type=$attn_type \
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
        done
    done
done
deactivate