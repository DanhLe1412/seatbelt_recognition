import argparse

import os
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout-ratio', type=float,default=0.5, help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--mode', type=int, default=0, help="0: train from scratch, 1: load checkpoint")
    parser.add_argument('--checkpoint-path',default="checkpoint_path/network.ckpt",help="Checkpoint path")
    parser.add_argument('--dir-save-model',default='saved_model',help='directory will save model after training')
    parser.add_argument('--dirTrain', default=None, help="Data to train model, Ex: '/datasets/trains' ")
    parser.add_argument('--dirValid', default=None, help="Data to Valid in train step, Ex: '/datasets/validation'")
    args = parser.parse_args()

    path = 'checkpoint_path'
    checkpoint_path = 'checkpoint_path/vgg16.ckpt'
    if os.path.exists(path) == False:
        os.mkdir(path)    

   


if __name__=="__main__":
    main()
