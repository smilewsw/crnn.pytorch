import os
import os.path as osp


def combine():
    result_dir = '/home/hezheqi/data/coco_text/submit/test_152_32w_poly'
    ocr_res = '/home/hezheqi/Project/crnn.pytorch/coco_e2e.txt'
    out_dir = '/home/hezheqi/data/coco_text/submit/test_152_32w_e2e'
    if not osp.isdir(out_dir):
        os.minor(out_dir)
    word_dic = {}
    with open(ocr_res) as fin:
        for line in fin:
            info = line.strip().split(',')
            if len(info) != 2:
                continue
            name, word = info
            name_split = name.split('_')
            name = str(int(name_split[-2])) + '_' + name_split[-1]
            word_dic[name] = word
    for f in os.listdir(result_dir):
        if 'txt' not in f:
            continue
        with open(osp.join(result_dir, f)) as fin:
            fout = open(osp.join(out_dir, f), 'w')
            for i, line in enumerate(fin):
                line = line.strip()
                name = f.split('_')[1].split('.')[0]
                name = name + '_' + str(i + 1)
                if name in word_dic:
                    fout.write('{},{}\n'.format(line, word_dic[name]))
                else:
                    fout.write('{},{}\n'.format(line, '###'))

if __name__ == '__main__':
    combine()

