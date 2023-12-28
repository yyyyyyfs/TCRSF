import os
import random
import shutil

img_dir = '/home/yfs/TCRSF/project/code/data/smoking/images'

save_train_file = os.path.join('/your/save/path','train.txt')
save_val_file = os.path.join('/your/save/path','val.txt')

img_list= os.listdir(img_dir)

class_folders = ['face', 'smoke', 'phone', 'drink']
train_img_list = []

num  = len(img_list)
train_list = random.sample(range(num),int(0.8*num))

for i,img in enumerate(img_list):
    if i in train_list:
        img_train_path = os.path.join(img_dir,img)
        train_file = open(save_train_file,'a')    
        train_file.write(img_train_path + '\n')
    else:
        img_val_path = os.path.join(img_dir,img)
        val_file = open(save_val_file,'a')
        val_file.write(img_val_path + '\n')

print("over,monster!")