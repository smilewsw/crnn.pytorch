from scipy import misc
import os
import os.path as osp
import re
import lmdb

def solve(tp):
    base_dir = '/home/hezheqi/data/coco_text/{}'.format(tp)
    fout = open(osp.join(base_dir, 'img_gt.txt'), 'w')
    img_out_dir = osp.join(base_dir, 'cropped_img')
    if not osp.isdir(img_out_dir):
        os.mkdir(img_out_dir)
    pattern = re.compile(r'\w+')
    with open(osp.join(base_dir, 'img_list.txt')) as fin:
        for line in fin:
            line = line.strip()
            gts = open(osp.join(base_dir, 'gt_box', line + '.txt')).read().strip().split('\n')
            im = misc.imread(osp.join(base_dir, 'img', line + '.jpg'))
            width, height = im.shape[:2]
            for i, gt_str in enumerate(gts):
                gt = gt_str.strip().split('\t')
                if len(gt) <= 1:
                    continue
                x, y, w, h = list(map(int, gt[:4]))
                x1 = max(0, x - 2)
                y1 = max(0, y - 2)
                x2 = min(x + 4, width)
                y2 = min(y + 4, height)
                word = ''.join(pattern.findall(gt[-1].lower()))
                # img[y: y + h, x: x + w]
                cropped = im[y1: y2, x1: x2]
                name = '{}_{:03d}.jpg'.format(line, i)
                misc.imsave(osp.join(img_out_dir, name), cropped)
                fout.write('{} {}\n'.format(name, word))


if __name__ == '__main__':
    for tp in ['train', 'valid']:
        solve(tp)
