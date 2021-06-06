import os
import argparse


def transform_dataset(label_path, fake_dir, real_dir):
    f = open(label_path, 'r')
    dataset_path = os.path.dirname(label_path)
    # print(dataset_path)
    Lines = f.readlines()
    for line in Lines:
        label = line.split()[1]
        path = dataset_path + "/"+ line.split()[0]
        base_name = line.split()[0].split("/")[1]+"_"+line.split()[0].split("/")[2]
        # print(path)
        # print(label)
        # print(base_name)
        if label == 0:
            print("cp "+path+" "+fake_dir+"/"+base_name)
            os.system("cp "+path+" "+fake_dir+"/"+base_name)
        else:
            print("cp "+path+" "+real_dir+"/"+base_name)
            os.system("cp "+path+" "+real_dir+"/"+base_name)
    f.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",   "--label_txt", help="Path to train_label.txt", required=True)
    parser.add_argument("-f",   "--fake_dir", help="Path to directory to store fake images", required=True)
    parser.add_argument("-r",   "--real_dir", help="Path to directory to store real images", required=True)
    
    args = parser.parse_args()
    if not os.path.exists(args.label_txt):
        print("train_label.txt path does not exist!")
        exit(0)
    if not os.path.exists(args.fake_dir):
        os.makedirs(args.fake_dir)
    if not os.path.exists(args.real_dir):
        os.makedirs(args.real_dir)
        
    transform_dataset(args.label_txt, args.fake_dir, args.real_dir)