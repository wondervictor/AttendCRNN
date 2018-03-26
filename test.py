# -*- coding: utf-8 -*-
from collections import OrderedDict
import torch
import argparse
import torch.utils.data
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
import utils
import dataset

import models.crnn as crnn
import models.attend_crnn as attend_crnn


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, help='Trained model path')
parser.add_argument('--use_attention', action='store_true', help='Use Attention')
parser.add_argument('--data_dir', required=True, help='path to dataset')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')


nc = 1
num_classes = 37

opt = parser.parse_args()

if opt.use_attention:
    crnn = attend_crnn.AttendCRNN(
        nc=nc,
        num_class=num_classes,
        hidden_size=opt.nh,
        use_cuda=opt.cuda
    )
else:
    crnn = crnn.CRNN(
        imgH=opt.imgH,
        nc=nc,
        nclass=num_classes,
        nh=opt.nh
    )

state_dict = torch.load(opt.model_path)
state_dict_rename = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    state_dict_rename[name] = v
crnn.load_state_dict(state_dict_rename)

print("Load Trained Model Finished!")

image = torch.FloatTensor(opt.batch_size, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batch_size * 5)
length = torch.IntTensor(opt.batch_size)


test_dataset = dataset.IIIT5k(root=opt.data_dir, training=False,)

nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.StrLabelConverter(opt.alphabet)
criterion = CTCLoss()

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)


def test(net, _dataset, criterion, save_attention=False):

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        _dataset,
        batch_size=opt.batch_size,
        num_workers=int(opt.workers),
        collate_fn=dataset.AlignCollate(img_height=opt.imgH, img_weight=opt.imgW, keep_ratio=opt.keep_ratio)
    )
    val_iter = iter(data_loader)

    n_correct = 0
    loss_avg = utils.Averager()

    result_file = open('result/{}_test_result.txt'.format('crnn_attention' if opt.use_attention else 'crnn'), 'a+')
    for i in range(len(data_loader)):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.load_data(text, t)
        utils.load_data(length, l)

        preds, atten_energy = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
        utils.plot_attention(atten_energy.cpu().data.numpy(), 'output_attention', '%s' % i)
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True) # [:opt.n_test_disp]

        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            result_file.write('%-20s => %-20s, gt: %-20s\n' % (raw_pred, pred, gt))

    result_file.close()
    accuracy = n_correct / float(len(data_loader) * opt.batch_size)

    print("[Test Result] Loss: {} Accuracy: {}".format(loss_avg.val(), accuracy))


test(crnn, test_dataset, criterion)
