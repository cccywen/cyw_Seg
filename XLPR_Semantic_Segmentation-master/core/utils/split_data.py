import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 10%的数据划分到验证集
    split_rate = 0.1

    # 指向flower_photos文件夹
    cwd = os.getcwd()  # 返回当前工作目录
    data_root = os.path.join(cwd, "dataset")
    origin_potsdam_path = os.path.join(cwd, "potsdam")
    assert os.path.exists(origin_potsdam_path)

    print("------------------------------------------")
    print(origin_potsdam_path)

    # 建立保存训练集的文件夹
    train_root = os.path.join(cwd, "train")
    mk_file(train_root)

    # 建立保存验证集的文件夹
    val_root = os.path.join(cwd, "val")
    mk_file(val_root)


    cla_path = os.path.join(origin_potsdam_path)
    images = os.listdir(cla_path)
    num = len(images)
    # 随机采样验证集的索引
    eval_index = random.sample(images, k=int(num*split_rate))
    for index, image in enumerate(images):
        if image in eval_index:
            # 将分配至验证集中的文件复制到相应目录
            print(image)
            # image_path = os.path.join(cla_path, image)
            # new_path = os.path.join(val_root)
            # copy(image_path, new_path)
        # else:
        #     # 将分配至训练集中的文件复制到相应目录
        #     image_path = os.path.join(cla_path, image)
        #     new_path = os.path.join(train_root)
        #     copy(image_path, new_path)
        # print("\rprocessing [{}/{}]".format(index+1, num), end="")  # processing bar
    print()

    print("processing done!")


if __name__ == '__main__':
    main()
