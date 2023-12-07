export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./checkpoint_pose_temporal_v7"
export CONTROLNET_MODEL="/raid/home/vibhu20150/litreview/adl-proj/train_controlnet/checkpoint_pose_finetuned_1_epoch"
export HOST_NODE_ADDR=localhost:50151
export CUDA_VISIBLE_DEVICES=4

torchrun --nproc-per-node=1 \
 --rdzv-backend=c10d \
 --rdzv-endpoint=$HOST_NODE_ADDR \
 train_controlnet_temporal.py --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=nin-ran-jan/vid360 \
 --resolution 512 184 \
 --controlnet_model_name_or_path=$CONTROLNET_MODEL \
 --validation_image "./validation_images/IMAGE0000.jpg" "./validation_images/IMAGE0001.jpg" "./validation_images/IMAGE0002.jpg" "./validation_images/IMAGE0003.jpg" "./validation_images/IMAGE0004.jpg" \
 --validation_prompt "Man with short black hair wearing a full sleeved red and black checked formal shirt along with black formal pants. paired with white and black striped socks and grey  and white shoes.he is also wearing black mask.carrying a beige backpack." "A man with black hair wearing black cap and Black headphones and wearing white half sleeve shirt and black jeans pant and wearing casual shoes of black and blue combination and in left hand wearing watch and holding mobile in his right hand and carrying backpack.\n" "A lady with black hair tied in a ponytail wearing a cream cap and a blue mask, a black white and grey textured tank top, black 3/4th joggers, black socks, blue and white sports shoes, a grey waist pack , wrist band and a plastic bag in hand." "A man with black short-length hair wearing  light grey cap , dark black half sleeve t-shirt with dark black jeans and casual shoes of white and dark black combination ." "A man with dark black hairs wearing dark black sweat shirt with hoodie topped upon whiteish grey t-shirt and dark black loose fit jeans and wearing  dark black belt slippers." \
 --learning_rate=1e-6 \
 --train_batch_size=1 \
 --train_frame_size=12 \
 --num_train_epochs=40 \
 --checkpointing_steps=100
