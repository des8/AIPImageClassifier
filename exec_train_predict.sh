# execute train.sh
#python train.py  --arch 'vgg19' --gpu True --epochs 10 --hidden_units 512 --learning_rate 0.001

python predict.py --image_path 'flowers/test/95/image_07505.jpg' --top_k 5 --gpu True