import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os.path as osp

import models.crnn as crnn

tp = 'val'
model_path = './data/crnn.pth'
data_base_dir = '/home/hezheqi/data/coco_text/words/{}'.format(tp)
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256, 1).cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))

result = open("{}.txt".format(tp), "w")

with open(osp.join(data_base_dir, 'img_list.txt')) as fin:
    for i, name in enumerate(fin):
        name = name.strip()
        img_path = osp.join(data_base_dir, 'img', name+'.jpg')
        image = Image.open(img_path).convert('L')
        image = transformer(image).cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        model.eval()
        preds = model(image)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        # print('%-20s => %-20s' % (raw_pred, sim_pred))
        result.write('{},{}\n'.format(name, sim_pred))
        print(i, end='\r')
