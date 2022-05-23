import os

if __name__ == '__main__':
    o_path = os.getcwd()
    print(o_path)
    par_path = os.getcwd().split("slns")[0]
    print("par_path" + par_path)
    i = 0
    print(par_path+"data\\Potsdam\\train.txt")
    # f = open(o_path+"data\\Potsdam\\train.txt", "w")

    # while i < 300:
    #     f.write('F:\data\Potsdam\images\{}.jpg'.format(i))
    #     f.write('\n')
    #     i += 1
    #
    # f.close()
