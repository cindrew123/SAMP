import argparse
import os
import itertools
import copyreg
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable, grad
import torch.optim as optim
# import scipy.misc
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score

from tensorboardX import SummaryWriter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torchvision import transforms
#import warnings
# xianshi
 
plt.switch_backend('agg')

import pickle
import cv2
import sys

from deeplab.seg_style_adapt_eye_model import VisionTransformer as ViT_seg
from deeplab.seg_style_adapt_eye_model import CONFIGS as CONFIGS_ViT_seg

from deeplab.seg_style_adapt_eye_model import diceCoeffv2, SoftDiceLoss, Res_Deeplab, Discriminator, Discriminator_NumClass_2
from deeplab.loss import CrossEntropy2d
from deeplab.datasets import VOCDataSet, VOCDataTestSet
from deeplab.isprs import ISPRS
from deeplab.isprs_vai import ISPRS_VAI
from deeplab.isprs_pot_irrg import ISPRS_POT_IRRG
from deeplab.isprs_pot_rgb import ISPRS_POT_RGB

import random
import timeit
import time

# style
import models.transformer as st_transformer
import models.StyTR  as StyTR

from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from PIL import Image
from collections import OrderedDict


# cut
from add_newest_tool.options.train_options import TrainOptions
from add_newest_tool.models.SAMP_model import SAMPModel


save_style_img_path = "/home/vipuser/trans/add_newest_tool/save_style_img"
save_cut_img_path = "/home/vipuser/trans/add_newest_tool/save_cut_img"


start = timeit.default_timer()

# IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
IMG_MEAN_BEIJING = np.array((82.90118227, 99.76638055, 96.33826206), dtype=np.float32)
IMG_MEAN_ISPRS = np.array((79.78101224, 81.65397782, 120.85911836), dtype=np.float32)


BATCH_SIZE = 2
NUM_CLASSES = 6

# DATA_DIRECTORY = '../../datasets/VOCdevkit/VOC2012'
# DATA_LIST_PATH = '../../datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
# NUM_CLASSES = 21
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4

MOMENTUM = 0.9
#NUM_STEPS = 10
NUM_STEPS =300
POWER = 0.9
RANDOM_SEED = 1234
#########
#Invocation_model = "True"
Invocation_model = "False"

Pre_Style_model = "True"
#Pre_Style_model = "False"

Pre_cut_model = "True"
#Pre_cut_model = "False"
epoch_to_load = "initial"


RESTORE_FROM = '/home/vipuser/trans/best_model/38/initial_seg.pth'
#RESTORE_FROM = '../datasets/target_potirrg_to_vai24000.pth'

PRE_STYLE_RESTORE_FROM = '/home/vipuser/trans/style_tool/save_style_model/style10000.pth'

SAVE_NUM_IMAGES = 2
#SAVE_PRED_EVERY = 5
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots1/'
WEIGHT_DECAY = 0.0005

SNAPSHOT_STYLE_DIR= './style_tool/save_style_model/'
SNAPSHOT_CUT_DIR= './add_newest_tool/save_cut_model/'


PLOT_STYLE_DIR= './style_tool/save_log/'
PLOT_CUT_DIR= './add_newest_tool/save_log/'

seg_model = 0 # 1-->multi, 0 --> only trans16

# seg_input = 121  # # 121-->{p1,p2}{p1'},11-->{ps1,ps1'}, 12 --> {ps1',ps2}, 120--> {ps1,ps2}, 110--> {ps1,ps1}，1212-> {ps1‘,ps2’}， 0-->{ps1,ps2}

trans_style_open = 0 # 1--> have trans style

# para of style control
Style_Control_Category = "Semantic_Awareness_Multiplication" #  "Semantic_Awareness_Multiplication";"Semantic_Awareness_Addition"; "Attention_Semantic_Awareness"
style_Innovation = 0# 1:OPEN the trans_style in cut,0 no. if change this , have to change this value in network.py and CUT_MODEL!!!!!
Style_gradient_control = 1 # 1:OPEN the Style_gradient_control
Style_Gradient_Control_Category = "attention" # "attention":Attention-map guided style gradient control."Interpolation":Simple gradient control
# Style_Gradient_Control_alpha = 0.5 #0 means the style is closer to style_A, 1 means the style is closer to style_B

# Cross_modal Cross_modal, seg_style_adapt_eye_model has centre_open set
centre_open=0 # Opening up cross-modal connections between style and semantic segmentation



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-d", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training epoches.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    # add VisionTransformer
    parser.add_argument('--vit_name', type=str,
                        default='ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--img_size', type=int,
                        default=512, help='input patch size of network input')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
                    
    parser.add_argument('--multi_trans_name', type=str,
                        default='basical-transblock', help='select one multi trans model')
    # set the depth of the multitrans
    parser.add_argument('--multi_trans_depth', type=str,
                        default='3', help='select multi trans depth')

    # add style translate
    parser.add_argument('--vgg', type=str, default='/home/vipuser/trans/style_tool/vgg_normalised.pth')
    parser.add_argument('--decoder_path', type=str, default='/home/vipuser/trans/style_tool/pre_decoder.pth')
    parser.add_argument('--Trans_path', type=str, default='/home/vipuser/trans/style_tool/pre_transformer.pth')
    parser.add_argument('--embedding_path', type=str, default='/home/vipuser/trans/style_tool/pre_embedding.pth')

    parser.add_argument('--save_dir', default='./style_tool/save_experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./style_tool/save_logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=NUM_STEPS)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=7.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument("--snapshot_style_dir", type=str, default=SNAPSHOT_STYLE_DIR,
                        help="Where to save snapshots of the style model.")
    parser.add_argument("--plot_style_dir", type=str, default=PLOT_STYLE_DIR,
                        help="Where to save PLOT of the style model.")
    parser.add_argument("--my_pre_style_model", type=str, default=PRE_STYLE_RESTORE_FROM,
                        help="Where my pre model parameters from.")
    
    # add cut
    parser.add_argument("--snapshot_cut_dir", type=str, default=SNAPSHOT_CUT_DIR,
                        help="Where to save snapshots of the cut model.")

    parser.add_argument("--plot_cut_dir", type=str, default=PLOT_CUT_DIR,
                        help="Where to save PLOT  of the CUT model.")                                            
    

                    
                    
    return parser.parse_args()


args = get_arguments()


def get_alpha_linear(current_epoch, total_epochs):
    return min(1.0, current_epoch / total_epochs)



def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1, and K is the number of segmented class.
    eg:
    palette:[[0],[1],[2],[3],[4],[5]]
    """
    semantic_map = []
    for colour in palette:
        #print mask
        mask2 = mask.data.cpu().numpy()
        equality = np.equal(mask2, colour)
        #print equality
        #class_map = np.all(equality, axis=-1)
        #print class_map
        semantic_map.append(equality)
        #print sum(equality)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    #print semantic_map
    return semantic_map

# imgt from target , imgs from source
# xiugai  butong
def loss_global_style(imgtrain, imgtest):
    # style_loss = loss_slobal_style(output_train_img, output_test_img)
    # output_train_img ge poti map ,output_test_img 1 VAI
    imgtrain1 = imgtrain[0, ...]
    # imgtrain1 (256L, 64L, 64L),256
    imgtest = imgtest[0, ...]
    loss_style = 0
    # imgtest (256L, 64L, 64L)

    for i in range(6):
        imgtr = (imgtrain1[i, ...]).cpu().detach().numpy()
        imgte = (imgtest[i, ...]).cpu().detach().numpy()
        meantr = np.mean(imgtr)
        meante = np.mean(imgte)
        vartr = np.std(imgtr)
        varte = np.std(imgte)
        meansty = float(meante - meantr)
        varsty = float(varte - vartr)

        loss_style = loss_style + abs(meansty) + abs(varsty)

    loss_style = loss_style/6
    loss_style = torch.tensor(loss_style, requires_grad = True).cuda()
    return loss_style

def loss_centre(centre_features, seg_layer):

    criterion = torch.nn.functional.mse_loss(centre_features, seg_layer)

    return criterion

def loss_calc_G(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # label = Variable(label.long()).cuda()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()

    return criterion(pred, label)


def loss_calc_D(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    # label = Variable(label.long()).cuda()
    criterion = torch.nn.BCELoss().cuda()

    return criterion(pred, label)


def res_loss_calc(real, fake):
    criterion = nn.L1Loss().cuda()

    return criterion(real, fake)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)
    # b.append(model.layer5)


    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            if i.requires_grad:
                yield i


def critic_params(model):
    b = []
    b.append(model.parameters())
    for j in range(len(b)):
        for i in b[j]:
            if i.requires_grad:
                yield i


def get_inf_iterator(data_loader):
    """Inf data iterator."""
    while True:
        for images, labels, size, name in data_loader:
            yield (images, labels, size, name)


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def calc_gradient_penalty(D, real_data, fake_data):
    """Calculatge gradient penalty for WGAN-GP."""
    alpha = torch.rand(real_data.size())
    # alpha = alpha.expand(args.batch_size, real_data.nelement() / args.batch_size).contiguous().view(real_data.size())
    alpha = alpha.cuda()

    interpolates = make_variable(alpha * real_data + ((1 - alpha) * fake_data))
    interpolates.requires_grad = True

    disc_interpolates, _ = D(interpolates)

    gradients = grad(outputs=disc_interpolates,
                     inputs=interpolates,
                     grad_outputs=make_cuda(
                         torch.ones(disc_interpolates.size())),
                     create_graph=True,
                     retain_graph=True,
                     only_inputs=True)[0]

    gradient_penalty = 10 * \
                       ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def adjust_learning_rate(optimizer, i_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def loss_plot(hist, path='Train_hist.png', model_name=''):
    x = hist['steps']

    y1 = hist['D_loss']
    y2 = hist['G_loss']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()


def loss_plot_sigle(hist, path='Train_hist.png', model_name=''):
    x = hist['steps']
    root = path
    for name, value in hist.items():
        if name == 'steps' or name == 'mIOUs' or name == 'Loss_train_pseudo' or name == 'Loss_test_pseudo':
            continue
        else:
            y = value
            plt.plot(x, y, label=name)
            plt.xlabel('Iter')
            plt.ylabel(name)
            plt.legend(loc=4)
            plt.grid(True)
            plt.tight_layout()

            loss_path = os.path.join(root, model_name + '_' + name + '.png')

            plt.savefig(loss_path)

            plt.close()


def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool
    from deeplab.metric2 import ConfusionMatrix
    #print 1
    #print data_list

    ConfM = ConfusionMatrix(class_num)
    
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()
    for m in m_list:
        ConfM.addM(m)
    aveJ, j_list, M, class_iou = ConfM.jaccard()
    
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list) + '\n')
            f.write(str(M) + '\n')
            f.write(str(class_iou) + '\n')

    return aveJ, class_iou

def get_train_cam_loss(pseudo, labels):  #Loss_train_pseudo = get_cam_loss(pseudo_avr_train, labels_train)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL).cuda()
    loss = criterion(pseudo, labels)
    return loss

def get_test_cam_loss(pseudo, labels):  #Loss_train_pseudo = get_cam_loss(pseudo_avr_train, labels_train)
    loss = F.multilabel_soft_margin_loss(pseudo, labels)
    return loss

def adjust_sty_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plot_and_save(x, y, title, ylabel, filename,plot_save_dir):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(plot_save_dir, filename))
    plt.close()

def test(model, step, testloader,content_layers_output,style_layers_output):
    model.eval()
    interp = nn.Upsample(size=(512, 512), mode='bilinear',align_corners=True)
    data_list = []

    for index, batch in enumerate(testloader):
        if index % 100 == 0:
            print('The''iter of ', index, 'processed')
        image, label, size, name = batch
        size = size[0].numpy()
        if seg_model == 1:
            with torch.no_grad():
              output, pseudo_avr, pseudo_vb, pseudo_multi = model(Variable(image).cuda())
        else:
            with torch.no_grad():
              output,centre_out,seg_out = model(Variable(image).cuda(),content_layers_output,style_layers_output)
        output = interp(output).cpu().data[0].numpy()

        output = output[:, :size[0], :size[1]]
        gt = np.asarray(label[0].numpy()[:size[0], :size[1]], dtype=int)

        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=int)
        gt = gt.flatten()
        output = output.flatten()
        class_F1 = f1_score(gt, output, average=None)
        mean_F1 = f1_score(gt, output, average='weighted')
        #print F1
        #print f1
        #bug

        # if index % 100 == 0 and index !=0:
        #    show_all(gt, output,name[0])
        data_list.append([gt.flatten(), output.flatten()])

    miou,class_iou = get_iou(data_list, args.num_classes,
                   './result1/target_potirrg_to_vai_change_numclass_2_softmax_bce_{}.txt'.format(step))
    return miou,class_iou,mean_F1,class_F1
    
## follow is style
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def save_tiff_image(tensor, filename):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = (image * 255).astype('uint8')

    image = image[..., [2, 1, 0]]

    pil_image = Image.fromarray(image)
    pil_image.save(filename, format='TIFF')
       

def remove_module_prefix(state_dict): # remove module word
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict



def main():
    """Create the model and start the training."""

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    gpu = args.gpu
    train_hist = {}
    train_hist['label_loss'] = []
    train_hist['mIOUs'] = []
    train_hist['steps'] = []

    train_hist['loss_G_GAN'] = []
    train_hist['loss_G'] = []
    train_hist['loss_D'] = []

    train_hist['semantic_loss'] = []




    evalu = {}
    evalu['mean_F1'] = []
    evalu['class_F1'] = []
    evalu['class_iou'] = []
    # cut loss
    loss_cut_D_values = []
    loss_cut_G_values = []
    cut_iterations = []
    # Create cut network.
    CUT_opt = TrainOptions().parse()
    CUT_MODEL = SAMPModel(CUT_opt)

    # load cut model parameters.
    if Pre_cut_model == "True":  # load my pre model
        print("---start load my pretrained cut modle---")
        CUT_MODEL.load_networks(epoch_to_load)

    if trans_style_open == 1: # 1--> have trans_style, 0-->no trans
        # style loss
        loss_content_values = []
        loss_style_values = []
        loss_identity1_values = []
        loss_identity2_values = []
        total_loss_values = []
        style_iterations = []

        # Create style network.
        vgg = StyTR.vgg
        vgg.load_state_dict(torch.load(args.vgg))
        vgg = nn.Sequential(*list(vgg.children())[:44])
        Style_decoder = StyTR.decoder
        Style_embedding = StyTR.PatchEmbed()
        style_transformer = st_transformer.Transformer()
        style_network = StyTR.StyTrans(vgg, Style_decoder, Style_embedding, style_transformer, args)

    if trans_style_open == 1: # 1--> have trans_style, 0-->no trans
        # load style model parameters.
        if Pre_Style_model == "True":  # load my pre model
            pretrained_dict  = torch.load(args.my_pre_style_model)
            pretrained_dict = remove_module_prefix(pretrained_dict)
            print("---start load my pretrained style modle---")
            style_network.load_state_dict(pretrained_dict)
    
        else:  # load the origin pre model
            new_state_dict = OrderedDict()
            state_dict = torch.load(args.decoder_path)
            for k, v in state_dict.items():
                # namekey = k[7:] # remove `module.`
                namekey = k
                new_state_dict[namekey] = v
            Style_decoder.load_state_dict(new_state_dict)
    
            new_state_dict = OrderedDict()
            state_dict = torch.load(args.Trans_path)
            for k, v in state_dict.items():
                # namekey = k[7:] # remove `module.`
                namekey = k
                new_state_dict[namekey] = v
            style_transformer.load_state_dict(new_state_dict)
    
            new_state_dict = OrderedDict()
            state_dict = torch.load(args.embedding_path)
            for k, v in state_dict.items():
                # namekey = k[7:] # remove `module.`
                namekey = k
                new_state_dict[namekey] = v
            Style_embedding.load_state_dict(new_state_dict)

        if trans_style_open == 1:
            print ("the trans style without open")
            style_network.train()
            style_network = nn.DataParallel(style_network)


    # Create seg network.
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    
    config_multi_trans = CONFIGS_ViT_seg[args.multi_trans_name]
    config_multi_trans.n_classes = args.num_classes
    config_multi_trans.n_skip = args.n_skip
    model = ViT_seg(config_vit, config_multi_trans, img_size=args.img_size, num_classes=config_vit.n_classes, multi_trans_depth = args.multi_trans_depth).cuda()


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    # load model parameters.
    if Invocation_model == "True":
        pretrained_dict = torch.load(args.restore_from)
        print("---start load pretrained seg modle---")
        model.load_state_dict(pretrained_dict)
    else:
        model.load_from(weights=np.load(config_vit.pretrained_path))

    # model_src.cuda()
    critic = Discriminator_NumClass_2(num_class=args.num_classes)
    critic.cuda()

    cudnn.benchmark = True
    # shuffle = False, the order of taking out images is the same.
    trainloader = data.DataLoader(ISPRS_POT_IRRG('fine', 'train', IMG_MEAN_BEIJING),
                                  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    testloader = data.DataLoader(ISPRS_VAI('fine', 'train', IMG_MEAN_ISPRS),
                                 batch_size=1, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    iters_of_epoch = len(testloader)
    target_testloader = data.DataLoader(
        ISPRS_VAI('fine', 'val', IMG_MEAN_ISPRS),
        batch_size=1, shuffle=False, pin_memory=True
    )

    train_loader = get_inf_iterator(trainloader)
    test_loader = get_inf_iterator(testloader)
    # optimizer_seg
    optimizer_d = optim.Adam(critic_params(critic),
                             lr=args.learning_rate_d, betas=(0.9, 0.99))

    if trans_style_open == 1: # 1--> have trans_style, 0-->no trans
        # optimizer_trans_style
        optimizer_st = optim.Adam([
            {'params': style_network.module.transformer.parameters()},
            {'params': style_network.module.decode.parameters()},
            {'params': style_network.module.embedding.parameters()},
        ], lr=args.lr)



    interp = nn.Upsample(size=input_size, mode='bilinear',align_corners=True)

    # interp = nn.Upsample(size=input_size, mode='bilinear')

    # num_images = len(trainloader)

    max_miou = 0
    c_iou = []
    m_F1 = 0
    c_F1 = []
    best_iter = 0

    num_images = len(testloader)
    criterion = nn.CrossEntropyLoss()
    lamda = 0.1

    writer = SummaryWriter(log_dir='./logs/target_p2v')
    base_lr = args.base_lr
    max_iterations = args.num_steps

    for step in range(args.num_steps):
        torch.autograd.set_detect_anomaly(True)

        if step <= 50000:
            seg_input = 121  # 121-->{p1,p2}{p1'},11-->{ps1,ps1'}, 12 --> {ps1',ps2}, 120--> {ps1,ps2}, 110--> {ps1,ps1}，1212-> {ps1‘,ps2’}， 0-->{ps1,ps2}

        elif step > 20000 & step<= 40000:
            seg_input = 121

        elif step > 50000 :
            seg_input = 120

        if trans_style_open == 1: # 1--> have trans_style, 0-->no trans
            if step < 1e4: 
                warmup_learning_rate(optimizer_st,iteration_count=step)
            else:
                adjust_sty_learning_rate(optimizer_st, iteration_count=step)

        time_start = time.time()
        model.train()
        critic.train()
        optimizer_seg = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

        iter_num = 0

        # train F

        # don't accumulate grads in D
        for param in critic.parameters():
            param.requires_grad = False

        adjust_learning_rate(optimizer_d, step)
        optimizer_d.zero_grad()

        batch_train = next(train_loader)
        batch_test = next(test_loader)
        images_train, labels_train, _, name_train = batch_train
        images_test, labels_test, _, name_test = batch_test
        images_train = Variable(images_train, requires_grad=False).cuda(args.gpu)
        images_test = Variable(images_test, requires_grad=False).cuda(args.gpu)
        labels_train = Variable(labels_train.long()).cuda(args.gpu)
        # compare CAM and real img
        print ("data is pori train ",name_train)
        print ("data is vai train ",name_test)

        # CUT CUT_opt
        content_images = images_train[0,:]
        PTS_labels = labels_train[0, :]
        style_images = images_test
        if step == 0:
        # each img all be transfer
            CUT_MODEL.set_input(content_images, style_images, PTS_labels)
            CUT_MODEL.data_dependent_initialize(content_images, style_images)
            CUT_MODEL.setup(CUT_opt)
            CUT_MODEL.parallelize()
        CUT_MODEL.set_input(content_images, style_images, PTS_labels)
        CUT_MODEL.forward()
        # save cut img
        if style_Innovation == 1:
            #fake_B,fake_B_tran,fake_B2A ,fake_B2A_tran = CUT_MODEL.get_generated_image(style_Innovation)
            fake_B, fake_B_tran, memory_out  = CUT_MODEL.get_generated_image(style_Innovation)
        else:
            fake_B = CUT_MODEL.get_generated_image(style_Innovation)
        #cut_visuals = fake_B
        if style_Innovation == 1:
            cut_visuals = CUT_MODEL.Semantic_perception(fake_B,fake_B_tran, style_images, Style_Control_Category)
        # cut_visuals_Contr = CUT_MODEL.Contrastive_Learning_Reinforcement_Learning(fake_B, fake_B_tran, fake_B2A, fake_B2A_tran)
        else:
            cut_visuals = fake_B
        if Style_gradient_control == 1:
            # Style_Gradient_Control_alpha = get_alpha_linear(step, args.num_steps)
            Style_Gradient_Control_alpha = 0
            cut_visuals = CUT_MODEL.Style_Gradient(cut_visuals, content_images, style_images, Style_Gradient_Control_Category,Style_Gradient_Control_alpha )
            cut_visuals = CUT_MODEL.structure_guided_style_fusion(content_images, cut_visuals, threshold=120,
                                                                  blend_strength=0.4)

        source_name = f"{save_cut_img_path}/cut_image_source_{(step )}.tif"
        save_tiff_image(content_images, source_name)

        style_name = f"{save_cut_img_path}/cut_image_style_{(step )}.tif"
        save_tiff_image(style_images, style_name)

        cut_out_name_fuse = f"{save_cut_img_path}/cut_image_{(step )}.tif"
        save_tiff_image(cut_visuals, cut_out_name_fuse)

        # Semantic Segmentation Network！！！！！！！！！！！！！
        images_train= cut_visuals


        content_layers_output, style_layers_output = CUT_MODEL.get_style_layers()
        if seg_model == 1:#  trans multi
            output_train, pseudo_avr_train, pseudo_vb_train, pseudo_multi_train = model(images_train)
            output_test, pseudo_avr_test, pseudo_vb_test, pseudo_multi_test  = model(images_test)

            labels_train = labels_train.to(dtype=torch.long)
            Loss_train_pseudo = get_train_cam_loss(pseudo_avr_train, labels_train)  #Loss_train_pseudo = tensor(1.7918, device='cuda:0', grad_fn=<NllLoss2DBackward>)
            pseudo_multi_test = pseudo_multi_test.squeeze()
            #pseudo_multi_test = pseudo_multi_test.to(dtype=torch.long)
            Loss_test_pseudo = get_test_cam_loss(pseudo_vb_test.squeeze(), pseudo_multi_test.squeeze()) # tensor(0.6937, device='cuda:0', grad_fn=<MeanBackward0>)
        else:
            output_train,centre_train,seg_train = model(images_train,content_layers_output,style_layers_output)
            labels_train = labels_train.to(dtype=torch.long)
            labels_train = labels_train[0, ...].unsqueeze(0)
            Loss_train_pseudo = 0  #Loss_train_pseudo = tensor(1.7918, device='cuda:0', grad_fn=<NllLoss2DBackward>)
            pseudo_multi_test = 0
            #pseudo_multi_test = pseudo_multi_test.to(dtype=torch.long)
            Loss_test_pseudo = 0 # tensor(0.6937, device='cuda:0', grad_fn=<MeanBackward0>)

        output_train = interp(output_train)
        out_source_loss = loss_calc_G(output_train, labels_train)

        if centre_open == 1:
            train_centre_loss = loss_centre(centre_train,seg_train)
            test_centre_loss = loss_centre(centre_test, seg_test)
            total_centre_loss = train_centre_loss+test_centre_loss
            print ("total_centre_loss is ", total_centre_loss)
            f_d_loss = out_source_loss  + D_loss_f * 0.01 + loss_style * 0.00+ Loss_train_pseudo*0.1 + Loss_test_pseudo*0.1 + total_centre_loss*0.001
        else:
            f_d_loss = out_source_loss



        CUT_MODEL.optimize_parameters(f_d_loss)

        loss_G_GAN, semantic_loss, loss_G, loss_D = CUT_MODEL.output_loss()
        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer_seg.param_groups:
            param_group['lr'] = lr_


        for param in critic.parameters():
            param.requires_grad = True
        # plot cut loss
        train_hist['loss_G_GAN'].append(loss_G_GAN.data.cpu().numpy())
        train_hist['loss_G'].append(loss_G.data.cpu().numpy())
        train_hist['loss_D'].append(loss_D.data.cpu().numpy())
        train_hist['semantic_loss'].append(semantic_loss.data.cpu().numpy())
        if step % 200 == 0:
            cut_out_name = f"{save_cut_img_path}/cut_image_{(step)}.tif"
            save_tiff_image(cut_visuals, cut_out_name)
        # print(step,loss_cut_D,"-loss_cut_D:",loss_cut_G,"-loss_cut_G:")


        # update parameters
        train_hist['label_loss'].append(out_source_loss.data.cpu().numpy())
        train_hist['steps'].append(step)
        if seg_model == 1: # multi seg have pseudo
            train_hist['Loss_train_pseudo'].append(Loss_train_pseudo.data.cpu().numpy())
            train_hist['Loss_test_pseudo'].append(Loss_test_pseudo.data.cpu().numpy())
            print ('The ', step, 'iter ', 'of' '/', args.num_steps, \
                'completed, D_loss_d = ', D_loss_d.data.cpu().numpy(), \
                ' , D_loss_f = ', D_loss_f.data.cpu().numpy(), \
                ' , f_d_loss = ', f_d_loss.data.cpu().numpy(), \
                ' , label_loss=', out_source_loss.data.cpu().numpy(), \
                ' , dice_loss=', dice_loss.data.cpu().numpy(), \
                ' , Loss_train_pseudo=', Loss_train_pseudo.data.cpu().numpy(), \
                ' , Loss_test_pseudo=', Loss_test_pseudo.data.cpu().numpy(), \
                ' , global_style_loss= ', loss_style.data.cpu().numpy())

            In_fo = {
                'D_loss_d': D_loss_d.data.cpu().numpy(),
                'D_loss_f': D_loss_f.data.cpu().numpy(),
                'f_d_loss': f_d_loss.data.cpu().numpy(),
                'label_loss': out_source_loss.data.cpu().numpy(),
                'dice_loss': dice_loss.data.cpu().numpy(),
                'Loss_train_pseudo': Loss_train_pseudo.data.cpu().numpy(),
                'Loss_test_pseudo': Loss_test_pseudo.data.cpu().numpy(),
                'global_style_loss': loss_style.data.cpu().numpy()
            }
        else: #only trans16 no pseudo
            print('The ', step, 'iter ', 'of' '/', args.num_steps, \
                  ' , label_loss=', out_source_loss.data.cpu().numpy())

            In_fo = {
                'label_loss': out_source_loss.data.cpu().numpy(),
            }
        for tag, value in In_fo.items():
            writer.add_scalar(tag, value, step + 1)
            # save model and count the best iou
        if (step + 1) % args.save_pred_every == 0:
            miou,class_iou,mean_F1,class_F1 = test(model, step + 1, target_testloader,content_layers_output,style_layers_output)
            evalu['mean_F1'].append(mean_F1)
            evalu['class_F1'].append(class_F1)
            evalu['class_iou'].append(class_iou)
            train_hist['mIOUs'].append(miou)
            if max_miou < miou:
                max_miou = miou
                c_iou = class_iou
                m_F1 = mean_F1
                c_F1 = class_F1
                best_iter = step + 1
                print('taking model snapshot ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir,
                                                        'vit16' + str(
                                                            step + 1) + '.pth'))
                print('taking critic snapshot ...')
                torch.save(critic.state_dict(), osp.join(args.snapshot_dir,
                                                         'critic_vit16' + str(
                                                             step + 1) + '.pth'))
            # save cut model
            if (step + 1) % 10000 == 0:
                print('taking cut model snapshot ...')
                CUT_MODEL.save_networks(str(step + 1))




            writer.add_scalar('mIOUs', miou, step + 1)
            print ("max_miou is : ", max_miou, " best_iter is : ", best_iter, "Current iteration miou is :", miou , "class_iou is : ", c_iou)
            print ("mean_F1 is : ", m_F1, "class_F1 is :", c_F1)
            # print 'taking sources snapshot ...'
            # torch.save(model_src.state_dict(), osp.join(args.snapshot_dir,
            #                                        'sources_isprs_scenes_v5_' + str(
            #                                            step + 1) + '.pth'))
        time_end = time.time()
        time_c= time_end - time_start
        print('time cost', time_c, 's')
        loss_plot_sigle(train_hist, './loss_plot1', 'cut')

    np.savetxt('./target_result/only_loss.txt', train_hist['mIOUs'])
    np.savetxt('./target_result/mean_F1.txt', evalu['mean_F1'])
    np.savetxt('./target_result/class_F1.txt', evalu['class_F1'], fmt = '%s')
    np.savetxt('./target_result/class_iou.txt', evalu['class_iou'], fmt = '%s')
    
    #loss_plot_sigle(train_hist, './loss_plot', 'vit16')

    end = timeit.default_timer()
    print (end - start, 'seconds')
    writer.close()


if __name__ == '__main__':
    main()
