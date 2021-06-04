gdown --id 1jwXwx8LfnydU1RQzSNRiXPe8G4-wjJnk
unzip state-farm-distracted-driver-detection.zip data/
python train_siamese.py --batch-size 32 --lr 1e-4 --epochs 30 --weight-path model/siamese/weight --dirTrain data/train --dirValid None