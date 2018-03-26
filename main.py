from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset

import models.crnn as crnn
import models.attend_crnn as attend_crnn

parser = argparse.ArgumentParser()
parser.add_argument('--use_attention', action='store_true', help='use self attention')
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--display_interval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=5, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=5, help='Interval to be saved')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = 2313# random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.IIIT5k(root=opt.trainroot, training=True)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.RandomSequentialSampler(train_dataset, opt.batch_size)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    collate_fn=dataset.AlignCollate(img_height=opt.imgH, img_weight=opt.imgW, keep_ratio=opt.keep_ratio)
)
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=opt.batch_size,
#     shuffle=True, sampler=sampler,
#     num_workers=int(opt.workers),)
    # collate_fn=dataset.AlignCollate(img_height=opt.imgH, img_weight=opt.imgW, keep_ratio=opt.keep_ratio))

test_dataset = dataset.IIIT5k(root=opt.valroot, training=False,)

nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.StrLabelConverter(opt.alphabet)
criterion = CTCLoss()


def adjust_lr(optimizer, epoch):
    lr = opt.lr * (0.2 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        if param_group['lr'] <= 0.00001:
            lr = 0.00001
        param_group['lr'] = lr


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = attend_crnn.AttendCRNN(nc=nc, num_class=nclass, hidden_size=opt.nh, use_cuda=opt.cuda) \
    if opt.use_attention else crnn.CRNN(opt.imgH, nc, nclass, opt.nh)


crnn.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.load_state_dict(torch.load(opt.crnn))

if not os.path.exists('log/'):
    os.mkdir('log')

logger = utils.Logger(
    stdio=True,
    log_file='log/{}_log.txt'.format('crnn_attention' if opt.use_attention else 'crnn')
)

print(crnn)

image = torch.FloatTensor(opt.batch_size, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batch_size * 5)
length = torch.IntTensor(opt.batch_size)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.Averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, _dataset, criterion, max_iter=100):

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        _dataset,
        shuffle=True,
        batch_size=opt.batch_size,
        num_workers=int(opt.workers),
        collate_fn=dataset.AlignCollate(img_height=opt.imgH, img_weight=opt.imgW, keep_ratio=opt.keep_ratio)
    )
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.Averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.load_data(text, t)
        utils.load_data(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    # for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
    #     print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batch_size)
    logger.log('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def train_batch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.load_data(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.load_data(text, t)
    utils.load_data(length, l)
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


logger.log('starting to train')
for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = train_batch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.display_interval == 0:
            logger.log('[%d/%d][%d/%d] Loss: %f'
                       % (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

    if (epoch+1) % opt.valInterval == 0:
        val(crnn, test_dataset, criterion)

    # do checkpointing
    if (epoch+1) % opt.saveInterval == 0:
        torch.save(
            crnn.state_dict(), '{0}/netCRNN_{1}.pth'.format(opt.experiment, epoch))

    if (epoch + 1) % 5 == 0:
        adjust_lr(optimizer, epoch+1)
