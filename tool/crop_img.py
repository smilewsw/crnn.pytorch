#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import unicode_literals
from xml.dom.minidom import parse
import xml.dom.minidom
import os
import os.path as osp
import argparse
import shutil
import multiprocessing
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser("Evaluate the effectiveness of the results.")
    parser.add_argument('--out_dir', dest='out_dir',
                        default='/home/hezheqi/data/detext/cropped',
                        help='Path to gt_file')
    parser.add_argument('--data_dir', dest='data_dir',
                        default='/home/hezheqi/data/detext',
                        help='Path to dis_file')
    parser.add_argument('--name', dest='name',
                        default='test_1600_poly',
                        help='Path to dis_file')

    return parser.parse_args()

def process_one(img_path, result_path, out_dir):
    img = cv2.imread(img_path)
    with open(result_path) as fin:
        for i, line in enumerate(fin):
            info = line.strip().split()
            if len(info) < 8 or float(info[-1]) < 0.7:
                continue
            quad = list(map(int, info[1:9]))
            cord = np.float32(quad)
            pts_src = cord.reshape(4, 2)
            # pts_src = np.float32([[cord[0],cord[1]],[cord[2],cord[3]],[cord[4],cord[5]],[cord[6],cord[7]]])
            w = max(np.linalg.norm(pts_src[1] - pts_src[0]), np.linalg.norm(pts_src[3] - pts_src[2]))
            h = max(np.linalg.norm(pts_src[3] - pts_src[0]), np.linalg.norm(pts_src[2] - pts_src[1]))
            pts_dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            # print(pts_src, pts_dst)
            H, status = cv2.findHomography(pts_src, pts_dst)
            if H.shape != (3, 3):
                continue
            img_dst = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
            if int(h) <= 0 or int(w) <= 0:
                continue
            img_p = img_dst[0:int(h), 0:int(w)]
            sub_img_name = img_path.split('/')[-1][:-4] + '_' + str(i) + '.jpg'
            sub_img_path = osp.join(out_dir, sub_img_name)
            # print(sub_img_path)
            cv2.imwrite(sub_img_path, img_p)

def main(data_base_dir, tp, result_name, out_dir):
    p = multiprocessing.Pool(50)
    result_dir = osp.join(data_base_dir, 'result', result_name)
    with open(osp.join(data_base_dir, tp, 'img_list.txt')) as fin:
        for name in fin:
            name = name.strip()
            img_path = osp.join(data_base_dir, tp, 'img', name + '.jpg')
            result_path = osp.join(result_dir, name + '.txt')
            p.apply_async(process_one, args=(img_path, result_path, out_dir))
            # process_one(img_path, result_path, out_dir)
            # break
    p.close()
    p.join()


def txt2txt(gt_file, dis_file, img_dir, dis_dir):
    with open(gt_file, "r") as fres:
        with open(dis_file, "w") as fdis:
            lines = fres.readlines()
            len_num = len(lines)
            idx = 0
            while idx < len_num:
                line = lines[idx].strip().split(" ")
                img_path = os.path.join(img_dir, line[1])
                img = cv2.imread(img_path)
                word_num = int(line[2])
                print(line[0], img_path, word_num)
                idx += 1
                for i in range(word_num):
                    word_info = lines[idx].strip().split(" ")
                    word = word_info[0]
                    cord = np.float32(word_info[1:])
                    #print(word, cord)
                    pts_src = cord.reshape(4, 2)
                    #pts_src = np.float32([[cord[0],cord[1]],[cord[2],cord[3]],[cord[4],cord[5]],[cord[6],cord[7]]])
                    w = max(np.linalg.norm(pts_src[1] - pts_src[0]),np.linalg.norm(pts_src[3] - pts_src[2]))
                    h = max(np.linalg.norm(pts_src[3] - pts_src[0]),np.linalg.norm(pts_src[2] - pts_src[1]))
                    pts_dst = np.float32([[0,0],[w,0],[w,h],[0,h]])
                    #print(pts_src, pts_dst)
                    H, status = cv2.findHomography(pts_src, pts_dst)
                    img_dst = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
                    img_p = img_dst[0:int(h), 0:int(w)]
                    sub_img_name = line[1].replace('/', '_').replace('.', '_' + str(i) + '.')
                    sub_img_path = os.path.join(dis_dir, sub_img_name)
                    # print(sub_img_path)
                    fdis.write(sub_img_name + ',' + word + '\n')
                    cv2.imwrite(sub_img_path, img_p)
                    idx += 1
                idx += 1


if __name__ == '__main__':
    args = parse_args()

    print(args.out_dir, 'test', args.name, args.out_dir)
    # do evaluation
    # txt2txt(args.gt_file, args.dis_file, args.img_dir, args.dis_dir)
    main(args.data_dir, 'test', args.name, args.out_dir)
