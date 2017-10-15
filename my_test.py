import torch
from torch.autograd import Variable
import utils
import dataset
import os
from PIL import Image
from collections import OrderedDict

import models.crnn as crnn

model_path = '/home/wangsiwei/Projects/crnn.pytorch/expr/from_pretrain_sorted_length_15_500/netCRNN_2_500.pth'
#model_path = '/home/wangsiwei/Projects/crnn.pytorch/my_models/from_pretrain_0.00005_adam/netCRNN_17_500.pth'
#img_path = '/home/wangsiwei/ocr_data/COCO-Text-words-test/1000012.jpg'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
img_dir = '/home/wangsiwei/ocr_data/coco_text/test'
#img_dir = '/home/wangsiwei/ocr_data/icdar15/Challenge2_Test_Task3_Images'
#img_dir = '/home/wangsiwei/ocr_data/icdar135/Challenge2_Test_Task3_Images/img'
#img_dir = '/home/wangsiwei/ocr_data/COCO-Text-words-test'
output_file = '/home/wangsiwei/Projects/crnn.pytorch/coco_text_test_result_sorted_15_2.txt'

model = crnn.CRNN(32, 1, 37, 256, 1).cuda()
print('loading pretrained model from %s' % model_path)
state_dict = torch.load(model_path)
state_dict_rename = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    state_dict_rename[name] = v
model.load_state_dict(state_dict_rename)
#model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)
 
def predict_word(img_path):
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(img_path).convert('L')
    image = transformer(image).cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    
    model.eval()
    preds = model(image)
    
    _, preds = preds.max(2)
    preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred

filelist = os.listdir(img_dir)
filelist.sort(key=lambda x:int(x[5:-4]))
with open(output_file, 'w') as fp:
    for file in filelist:
        img_path = os.path.join(img_dir, file)
        word = predict_word(img_path)
        fp.write(file[:-4] + "," + word + "\n")

