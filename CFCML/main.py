import os
import torch
import time
import math
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from decimal import Decimal, ROUND_HALF_UP
from scheduler import GradualWarmupScheduler
from data_provider_men import TB_Dataset as TB_Dataset_men
from data_provider_derm7pt_2c import TB_Dataset as TB_Dataset_derm7pt
from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import auc as AUC
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, average_precision_score
from CFCML import CFCML

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--method_name', type=str, default='CFCML')
    parser.add_argument('--multi_gpu', type=int, default=0)
    parser.add_argument('--ifbalanceloader', type=int, default=0)
    parser.add_argument('--ifbalanceloader2', type=int, default=0) # Dataloader 一次性load batch_size个数据, 只对训练时有效
    parser.add_argument('--ifbalanceloader3', type=int, default=0) # Dataloader 一次性load num_classes个数据, 只对训练时有效
    parser.add_argument('--ifbalanceloader4', type=int, default=1) # Dataloader 一次性load num_classes个数据, 只对训练时有效, 并进行online aug
    parser.add_argument('--ifbalanceloader5', type=int, default=0)
    parser.add_argument('--ifoffline_data_aug', type=int, default=0)
    parser.add_argument('--ifonline_data_aug', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default='men')
    parser.add_argument('--img_size_men', type=int, default=128)
    parser.add_argument('--fold_train', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--train_epochs_abide', type=int, default=800)
    parser.add_argument('--early', type=int, default=400) # early stop
    parser.add_argument('--ifwarmup', type=bool, default=True)
    parser.add_argument('--run_type', type=str, default='train') # train, test, vis, heatmap, vis_space, heatmap_moe_weights, wilcoxon_test_get_pred, get_param
    parser.add_argument('--loss_type', type=str, default='ce')
    parser.add_argument('--testTraindata', type=int, default=0) # test_traindata
    parser.add_argument('--test_type', type=str, default='ori') # ori, ori_noise
    parser.add_argument('--batch_size_men', type=int, default=36)
    parser.add_argument('--batch_size_derm7pt', type=int, default=64)
    parser.add_argument('--train_lr_men', type=float, default=0.0005)
    parser.add_argument('--train_lr_derm7pt', type=float, default=0.0001)
    parser.add_argument('--cls_drop', type=float, default=0.5)
    parser.add_argument('--weight_decay_value', type=float, default=0.0001)
    parser.add_argument('--step_size', type=int, default=5, help='学习率下降策略，每隔多少个epoch对学习率进行下降')
    parser.add_argument('--lr_gamma', type=float, default=0.8, help='学习率下降策略，每隔step_size个epoch将学习率下降为之前的lr_gamma')
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--save_epoch_start', type=int, default=5)
    parser.add_argument('--ifclinical', type=int, default=1)
    parser.add_argument('--image_encoder', type=str, default='res_mamba')
    parser.add_argument('--clinical_encoder', type=str, default='clip')
    parser.add_argument('--ifshared_image_encoder', type=int, default=1) # 不同的模态图像使用共享权重的encoder提取特征，在image_encoder为swin_t时起作用
    parser.add_argument('--ifcie', type=int, default=1) # Crossmodal information enhancement
    parser.add_argument('--cie_att_type', type=str, default='mhca')
    parser.add_argument('--completion_type', type=str, default='concat')
    parser.add_argument('--num_latents_img', type=int, default=32)
    parser.add_argument('--num_latents_derm7pt', type=int, default=32)
    parser.add_argument('--num_latents_clinical', type=int, default=16)
    parser.add_argument('--ifdif_token_num', type=int, default=1)
    parser.add_argument('--ifmixup_class', type=int, default=0)
    #model
    parser.add_argument('--mamba_drop', type=float, default=0.5)
    parser.add_argument('--mamba_drop_path', type=float, default=0.5)
    parser.add_argument('--if_cls_token', type=float, default=0)
    parser.add_argument('--final_pool_type', type=str, default='mean')
    parser.add_argument('--multi_gpus', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--finetuning', type=int, default=0)
    parser.add_argument('--ft_rate', type=float, default=0.1)
    parser.add_argument('--w_main_cls', type=float, default=1)
    parser.add_argument('--w_loss_sam', type=float, default=0.03)
    parser.add_argument('--w_loss_up', type=float, default=1)
    parser.add_argument('--w_loss_cp', type=float, default=1)
    parser.add_argument('--CCRM_diff_loss_type', type=str, default='cs')
    parser.add_argument('--ifmultigranularity', type=int, default=1) # multi-granularity
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--CCRM_type', type=str, default='proto') # proto, targeted_proto, targeted_proxy
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--blocks', type=int, default=3) #很占内存空间
    #mamba
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--expand', type=int, default=1)

    return parser.parse_args()
args = get_parser()

train_epochs = args.train_epochs
args.vis_epoch = train_epochs

if args.dataset_name == 'men':
    args.image_encoder = 'res_mamba'
    modal_num = 4
    num_classes = 3
    train_lr = args.train_lr_men
    batch_size = args.batch_size_men
    batch = args.batch_size_men // num_classes # ifbalanceloader4
    number = 434
elif args.dataset_name == 'derm7pt':
    args.image_encoder = 'swin_t'
    args.finetuning = 1
    modal_num = 3
    num_classes = 2
    train_lr = args.train_lr_derm7pt
    batch_size = args.batch_size_derm7pt

    batch = args.batch_size_derm7pt // num_classes # ifbalanceloader4
    number = 256

display_num = int(math.floor(number/batch)//1.5) # ifbalanceloader4

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
np.random.seed(args.seed)  # Numpy module.
random.seed(args.seed)  # Python random module.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


cuda = True if torch.cuda.is_available() else False


def one_hot(x, class_count):
    return (torch.eye(class_count)[x,:]).cuda()


def one_hot_forauc(x, class_count):
    return np.eye(class_count)[x,:]


def val_derm7pt(model, dataloaders):
    correct = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    y_score = []
    y_true = []

    preds = []  # 记录每个病例的模型预测结果
    labels = []  # 记录每个病例的分类标签

    with torch.no_grad():
        for x, clinical, y, patient in dataloaders:

            x = x.type(torch.FloatTensor)
            x = x.cuda()  # if voting,[1,aug_num,2,128,128,24]; else,[2,128,128,24]
            clinical = clinical.cuda()
            y = y.cuda()
            clinical = clinical.cuda()
            labels.append(y.cpu().numpy()[0])

            if args.method_name == 'CFCML':
                out = model(x, clinical, y, run_type='test')
                out_pred = out['out_prediction']
            else:
                print('wrong method_name!')
                break

            out_pred = F.softmax(out_pred, dim=1)
            pred = out_pred.argmax(dim=1, keepdim=True)
            y_score.append(out_pred[0, 1].cpu().numpy())
            preds.append(pred.cpu().numpy()[0])
            y_true.append(y.cpu().numpy())

            correct += pred.eq(y.view_as(pred)).sum().item()
            if pred.view_as(y) == y == 1: TP += 1
            if (pred.view_as(y) == 1) and (y == 0): FP += 1
            if (pred.view_as(y) == 0) and (y == 1): FN += 1
            if pred.view_as(y) == y == 0: TN += 1

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)

    sen = Decimal(TP / (TP + FN)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
    spe = Decimal(TN / (TN + FP)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
    acc = Decimal((TP + TN) / (TP + FP + FN + TN)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
    g_mean = Decimal(((TP / (TP + FN)) * (TN / (TN + FP))) ** 0.5).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
    balance_acc = Decimal(balanced_accuracy_score(labels, preds)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
    auprc = Decimal(average_precision_score(labels, preds)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
    auc = Decimal(AUC(fpr, tpr)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
    mcc = Decimal(matthews_corrcoef(labels, preds)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)

    mean_metrics = (sen + spe + acc + g_mean + balance_acc +auprc + auc) / 7
    out_show = '(sen:{}, spe:{}, acc:{}, g_mean:{}, balance_acc:{}, auprc:{}, auc:{})'.format(str(sen), str(spe), str(acc), str(g_mean), str(balance_acc), str(auprc), str(auc))

    return {'mean_metrics': mean_metrics,
            'out_show': out_show,
            'sen': sen,
            'spe': spe,
            'acc': acc,
            'g_mean': g_mean,
            'balance_acc': balance_acc,
            'auprc': auprc,
            'auc': auc,
            'mcc': mcc}


def val(model,labels=None):
    model.eval()

    if args.dataset_name == 'men':
        test_set = TB_Dataset_men('test', use_path, clinical_path, num_classes, val=1, args=args)
    elif args.dataset_name == 'derm7pt':
        val_set = TB_Dataset_derm7pt('val', use_path, clinical_path, args=args)
        test_set = TB_Dataset_derm7pt('test', use_path, clinical_path, args=args)
    else:
        print('wrong dataset_name!')

    # 2class
    if args.dataset_name == 'derm7pt':
        out_val = val_derm7pt(model, DataLoader(val_set, batch_size=1, num_workers=args.num_workers))
        val_mean_metrics = out_val['mean_metrics']
        val_out_show = out_val['out_show']

        out_test = val_derm7pt(model, DataLoader(test_set, batch_size=1, num_workers=args.num_workers))
        test_out_show = out_test['out_show']

        return {'mean_metrics': val_mean_metrics,
                'out_show': val_out_show,
                'test_out_show':test_out_show,
                'sen': out_test['sen'],
                'spe': out_test['spe'],
                'acc': out_test['acc'],
                'g_mean': out_test['g_mean'],
                'balance_acc': out_test['balance_acc'],
                'auprc': out_test['auprc'],
                'auc': out_test['auc'],
                'mcc': out_test['mcc']}
    elif args.dataset_name == 'men':
        dataloaders = DataLoader(test_set, batch_size=1, num_workers=args.num_workers)

        all_preds = []  # 记录每个病例的模型预测结果
        all_preds_score = []
        labels = []  # 记录每个病例的侵袭和分级标签

        with torch.no_grad():
            for x, clinical, y, patient in dataloaders:

                x = x.type(torch.FloatTensor)
                x = x.cuda()
                clinical = clinical.cuda()
                y = y.cuda()
                labels.append(y.cpu().numpy()[0])

                if args.method_name == 'CFCML':
                    out = model(x, clinical, y, run_type='test')
                    out_pred = out['out_prediction']
                else:
                    print('wrong method_name!')
                    break

                out_pred = F.softmax(out_pred, dim=1)
                pred = out_pred.argmax(dim=1, keepdim=False)

                all_preds.append(pred.cpu().numpy()[0])
                all_preds_score.append(out_pred.cpu().numpy()[0])

        right_0 = 0
        right_1 = 0
        right_2 = 0
        preds = np.array(all_preds)

        for i in range(preds.shape[0]):
            if preds[i] == labels[i] and preds[i] == 0:
                right_0 += 1
            elif preds[i] == labels[i] and preds[i] == 1:
                right_1 += 1
            elif preds[i] == labels[i] and preds[i] == 2:
                right_2 += 1
        num_0 = len([i for i, x in enumerate(labels) if x == 0])
        num_1 = len([i for i, x in enumerate(labels) if x == 1])
        num_2 = len([i for i, x in enumerate(labels) if x == 2])

        all_preds_score_np = np.array(all_preds_score)
        if not np.isfinite(all_preds_score_np).all():
            print(all_preds_score_np)

        acc_g1 = Decimal(right_0 / num_0).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        acc_g2inv = Decimal(right_1 / num_1).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        acc_g2noninv = Decimal(right_2 / num_2).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        acc = Decimal(accuracy_score(labels,all_preds)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        weighted_f1 = Decimal(f1_score(labels,all_preds, average='weighted')).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        macro_f1 = Decimal(f1_score(labels,all_preds, average='macro')).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        auc = Decimal(roc_auc_score(one_hot_forauc(labels, num_classes), all_preds_score)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)

        mean_metrics = (acc+acc_g1+acc_g2inv+acc_g2noninv+weighted_f1+macro_f1+auc)/6
        out_show = '(acc:{}, acc_g1:{}, acc_g2inv:{}, acc_g2noninv:{}, weighted_f1:{}, macro_f1:{}, auc:{})'.format(str(acc),str(acc_g1),str(acc_g2inv),str(acc_g2noninv),str(weighted_f1),str(macro_f1),str(auc))
        return {'mean_metrics': mean_metrics,
                'out_show': out_show,
                'acc_g1':acc_g1,
                'acc_g2inv':acc_g2inv,
                'acc_g2noninv':acc_g2noninv,
                'acc':acc,
                'weighted_f1':weighted_f1,
                'macro_f1':macro_f1,
                'auc':auc}


def train_model_crossmodal(model, optimizer, train_sampler, dataload, scheduler, scheduler_warmup, num_epochs, loss_type):
    step_ls = []

    start_time = datetime.datetime.now()
    print('epoch start time: ', start_time)

    step_ls.append(use_path)
    step_ls.append(start_time)

    time_path = '{}-{}_{}-{}'.format(str(str(start_time).split(' ')[0]),str((str(start_time).split(' ')[1]).split(':')[0]),str((str(start_time).split(' ')[1]).split(':')[1]),str((str(start_time).split(' ')[1]).split(':')[2]))

    if args.dataset_name == 'derm7pt':
        weight_path = os.path.join('../weights', args.dataset_name, args.method_name, time_path)
    else:
        weight_path = os.path.join('../weights', args.dataset_name, args.method_name, time_path, str(fold))

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    best_val_result = 0
    mean_metrics = 0

    for epoch in range(num_epochs):
        model.train() # 每个epoch都会进行model.eval(), 因此需要每个epoch加上model.train()

        show_param_groups = 0
        print('\nEpoch {}/{}, lr:{}, time:{}\n{}'.format(str(epoch+1), str(num_epochs), str(optimizer.state_dict()['param_groups'][0]['lr']), str(datetime.datetime.now()), '-' * 60))

        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0

        for inputs, clinical_data, labels, patient in dataload:
            step += 1
            if args.dataset_name == 'men' or args.dataset_name == 'abide':
                if args.ifbalanceloader2:
                    inputs = torch.squeeze(inputs, 0)
                    clinical_data = torch.squeeze(clinical_data, 0)
                    labels = torch.squeeze(labels, 0)
                elif args.ifbalanceloader3 or args.ifbalanceloader4 or args.ifbalanceloader5:
                    inputs = inputs.reshape(-1, inputs.size(2), inputs.size(3), inputs.size(4), inputs.size(5))
                    if args.clinical_encoder == 'clip':
                        clinical_data = clinical_data.reshape(-1, clinical_data.size(2), clinical_data.size(3)) # [batch, token, dim]
                    else:
                        clinical_data = clinical_data.reshape(-1, clinical_data.size(2))
                    labels = labels.reshape(-1)
                    patient = [item for sublist in patient for item in sublist]
            elif args.dataset_name == 'derm7pt':
                if args.ifbalanceloader3 or args.ifbalanceloader4:
                    inputs = inputs.reshape(-1, inputs.size(2), inputs.size(3), inputs.size(4), inputs.size(5))
                    if args.clinical_encoder == 'clip':
                        clinical_data = clinical_data.reshape(-1, clinical_data.size(2), clinical_data.size(3))  # [batch, token, dim]
                    else:
                        clinical_data = clinical_data.reshape(-1, clinical_data.size(2))
                    labels = labels.reshape(-1)
                    patient = [item for sublist in patient for item in sublist]

            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            labels = labels.cuda()
            clinical_data = clinical_data.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            out = model(inputs, clinical_data, labels=labels, epoch=epoch)

            loss = out['loss']
            out_pred = out['out_prediction']

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if args.dataset_name == 'men':
                pred = out_pred.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                if step % display_num == 0:
                    right_0 = 0
                    right_1 = 0
                    right_2 = 0
                    for i in range(pred.shape[0]):
                        if pred[i] == labels[i] and pred[i] == 0:
                            right_0 += 1
                        elif pred[i] == labels[i] and pred[i] == 1:
                            right_1 += 1
                        elif pred[i] == labels[i] and pred[i] == 2:
                            right_2 += 1
                    num_0 = len([i for i, x in enumerate(labels) if x == 0])
                    num_1 = len([i for i, x in enumerate(labels) if x == 1])
                    num_2 = len([i for i, x in enumerate(labels) if x == 2])

                    if args.method_name == 'CFCML':
                        display_data = ' || loss:{} (main_cls_loss:{}, loss_sam:{}, loss_up:{}, loss_cp:{}) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}, 2:{}/{}]'.format(
                            str(loss.item()), str(out['main_cls_loss'].item()),
                            str(out['loss_sam'].item()),
                            str(out['loss_up'].item()),
                            str(out['loss_cp'].item()),
                            correct, batch_size, 100. * correct / batch_size, str(right_0),
                            str(num_0), str(right_1), str(num_1), str(right_2), str(num_2))
                    else:
                        display_data = ' || loss:{} (main_cls_loss:{}) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}, 2:{}/{}]'.format(
                            str(loss.item()), str(out['main_cls_loss'].item()), correct, batch_size, 100. * correct / batch_size,
                            str(right_0), str(num_0), str(right_1), str(num_1), str(right_2), str(num_2))

                    print(' ' * 5 + str(step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)
                    step_ls.append('epoch:' + str(epoch + 1) + ', lr: ' + str(optimizer.state_dict()['param_groups'][show_param_groups]['lr']) +'\n'+' ' * 5 + str(step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)

            elif args.dataset_name == 'derm7pt':
                pred = out_pred.argmax(dim=1, keepdim=True)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                if step % display_num == 0:
                    right_0 = 0
                    right_1 = 0

                    for i in range(pred.shape[0]):
                        if pred[i] == labels[i] and pred[i] == 0:
                            right_0 += 1
                        elif pred[i] == labels[i] and pred[i] == 1:
                            right_1 += 1

                    num_0 = len([i for i, x in enumerate(labels) if x == 0])
                    num_1 = len([i for i, x in enumerate(labels) if x == 1])

                    if args.method_name == 'CFCML':
                        display_data = ' || loss:{} (main_cls_loss:{}, loss_sam:{}, loss_up:{}, loss_cp:{}) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                            str(loss.item()), str(out['main_cls_loss'].item()), str(out['loss_sam'].item()), str(out['loss_up'].item()), str(out['loss_cp'].item()),
                            correct, batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1))
                    else:
                        display_data = ' || loss:{} (main_cls_loss:{}) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                            str(loss.item()), str(out['main_cls_loss'].item()), correct, batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1))

                    print(' ' * 5 + str(step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)

                    step_ls.append('epoch:' + str(epoch + 1) + ', lr: ' + str(optimizer.state_dict()['param_groups'][show_param_groups]['lr']) + str(step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)

        print(' ' * 5 + 'epoch ' + str(epoch + 1), ': loss is ' + str(epoch_loss / step)+',  ',datetime.datetime.now())

        val_result = val(model)
        mean = val_result['mean_metrics']
        out_show = val_result['out_show']

        if args.dataset_name == 'men' or args.dataset_name == 'abide':
            print(' ' * 5 + '----> ' + out_show)
            step_ls.append(' ' * 5 + '----> ' + out_show)
        if args.dataset_name == 'derm7pt':
            print(' ' * 5 + '----> val results:\n' + out_show)
            step_ls.append(' ' * 5 + '----> val results:\n' + out_show)
            print(' ' * 5 + '----> test results:\n' + val_result['test_out_show'])
            step_ls.append(' ' * 5 + '----> test results:\n' + val_result['test_out_show'])

        if mean > mean_metrics:
            best_val_result = val_result

            mean_metrics = mean
            if (epoch + 1) > args.save_epoch_start or args.dataset_name == 'brats' or 'mrnet' in args.dataset_name:
                if args.dataset_name == 'derm7pt':
                    torch.save(model.state_dict(), os.path.join(weight_path, 'train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                    torch.save(model.state_dict(), os.path.join(weight_path, 'train_weight_' + str(num_epochs) + '_' + str(batch_size) + '.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(weight_path, fold + '_train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                    torch.save(model.state_dict(), os.path.join(weight_path, fold + '_train_weight_' + str(num_epochs) + '_' + str(batch_size) + '.pth'))
                print(' ' * 5 + '---->  save weight of epoch {}!'.format(str(epoch + 1)))
                step_ls.append(' ' * 5 + '---->  save weight of epoch {}!'.format(str(epoch + 1)))

        if train_epochs == 50:
            if (epoch + 1) % 10 == 0 and (epoch + 1) > 30:
                if args.dataset_name == 'derm7pt':
                    torch.save(model.state_dict(), os.path.join(weight_path, 'train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(weight_path, fold + '_train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))

        if args.ifwarmup:
            scheduler_warmup.step()
        else:
            scheduler.step()

    result_ls = []
    result_ls.append(time_path)
    if args.dataset_name == 'men':
        result_ls.append(best_val_result['acc'])
        result_ls.append(best_val_result['acc_g1'])
        result_ls.append(best_val_result['acc_g2inv'])
        result_ls.append(best_val_result['acc_g2noninv'])
        result_ls.append(best_val_result['weighted_f1'])
        result_ls.append(best_val_result['macro_f1'])
        result_ls.append(best_val_result['auc'])

        result_ls = np.reshape(result_ls, (1, len(result_ls)))
        if not os.path.exists(test_result_path):
            head_ls = ['weight_date', 'acc', 'acc_g1', 'acc_g2inv', 'acc_g2noninv', 'weighted_f1', 'macro_f1', 'auc']
            head_ls = np.reshape(head_ls, (1, len(head_ls)))
            result_out = np.concatenate((head_ls, result_ls), axis=0)
        else:
            result_in = pd.read_csv(test_result_path, header=None)
            result_out = np.concatenate((result_in, result_ls), axis=0)
        result_out_pd = pd.DataFrame(result_out)
        result_out_pd.to_csv(test_result_path, header=False, index=False)
        print("Finish saving result_men.csv!")

    elif args.dataset_name == 'derm7pt':
        result_ls.append(best_val_result['sen'])
        result_ls.append(best_val_result['spe'])
        result_ls.append(best_val_result['acc'])
        result_ls.append(best_val_result['g_mean'])
        result_ls.append(best_val_result['balance_acc'])
        result_ls.append(best_val_result['auprc'])
        result_ls.append(best_val_result['auc'])

        result_ls = np.reshape(result_ls, (1, len(result_ls)))
        if not os.path.exists(test_result_path):
            head_ls = ['weight_date', 'Sen', 'Spe', 'Acc', 'G_mean', 'Ba_Acc', 'auprc', 'Auc']
            head_ls = np.reshape(head_ls, (1, len(head_ls)))
            result_out = np.concatenate((head_ls, result_ls), axis=0)
        else:
            result_in = pd.read_csv(test_result_path, header=None)
            result_out = np.concatenate((result_in, result_ls), axis=0)
        result_out_pd = pd.DataFrame(result_out)
        result_out_pd.to_csv(test_result_path, header=False, index=False)

        print("Finish saving result_derm7pt.csv!")

    end_time = datetime.datetime.now()
    print('\nbest_val_result: \n' + best_val_result['out_show'])
    step_ls.append('\nbest_val_result: \n' + best_val_result['out_show'])
    if args.dataset_name == 'derm7pt':
        print('best_test_result: \n' + best_val_result['test_out_show'])
        step_ls.append('best_test_result: \n' + best_val_result['test_out_show'])
    step_ls.append(end_time)
    step_ls.append('running use time: {}'.format(str(end_time-start_time)))
    step_ls_pd = pd.DataFrame(step_ls)

    if args.dataset_name == 'derm7pt':
        step_ls_pd.to_csv(os.path.join(weight_path, 'train_step_' + str(num_epochs) + '.csv'), index=False, header=False)
    else:
        step_ls_pd.to_csv(os.path.join(weight_path, fold + '_train_step_' + str(num_epochs) + '_' + str(batch_size) + '.csv'), index=False, header=False)


def model_select():
    if args.loss_type == 'ce':
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    else:
        print('wrong loss_type!')

    if args.method_name == 'CFCML':
        model = CFCML(args=args, criterion=criterion, num_classes=num_classes)
    else:
        print('model_select -- wrong method_name!')

    print('-'*30+'  '+args.method_name+'  '+'-'*30)

    return model


def train(batch_size, lr, saved_weight_path):
    model = model_select()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    print("Number of parameter: %.2fM" % (sum([np.prod(p.size()) for p in model_parameters]) / 1e6))
    model = model.cuda()

    model.train()

    if args.finetuning:
        base_parameters = []
        for pname, p in model.named_parameters():
            if 'swin_t' in pname:
                base_parameters.append(p)
        print('Num of base parameters: %.2fM' % (sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, base_parameters)]) / 1e6))
        base_parameters_id = list(map(id, base_parameters))
        new_parameters = list(filter(lambda p: id(p) not in base_parameters_id, model.parameters()))
        print('Num of dis parameters: %.2fM' % (sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, new_parameters)]) / 1e6))
        model_params = [{"params": base_parameters, "lr": train_lr * args.ft_rate},
                        {"params": new_parameters, "lr": train_lr}]

        optimizer = optim.Adam(model_params, lr=lr, weight_decay=args.weight_decay_value)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay_value)


    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_gamma)

    if args.ifwarmup:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epoch, after_scheduler=scheduler)
    else:
        scheduler_warmup=False

    if args.dataset_name == 'men':
        train_set = TB_Dataset_men('train', use_path, clinical_path, num_classes, val=0, args=args)
    elif args.dataset_name == 'derm7pt':
        train_set = TB_Dataset_derm7pt('train', use_path, clinical_path, args=args)

    dataloaders = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=args.num_workers, drop_last=True)
    train_sampler = False

    if args.method_name == 'CFCML':
        train_model_crossmodal(model, optimizer, train_sampler, dataloaders, scheduler, scheduler_warmup, train_epochs, args.loss_type)
    else:
        print('train -- wrong method_name!')


def test(weight_path, num_classes, weight_date):

    model = model_select()
    print('load data from {}'.format(str(weight_path)))
    model.load_state_dict(torch.load(weight_path))

    model = model.cuda()
    model.eval()

    if args.dataset_name == 'men':
        test_set = TB_Dataset_men('test', use_path, clinical_path, num_classes, val=1, args=args)
    elif args.dataset_name == 'derm7pt':
        test_set = TB_Dataset_derm7pt('test', use_path, clinical_path, args=args)
    else:
        print('Wrong dataset_name!')

    dataloaders = DataLoader(test_set, batch_size=1, num_workers=args.num_workers)
    if args.dataset_name == 'men':
        count = 0
        all_preds = []  # 记录每个病例的模型预测结果
        all_preds_score = []
        labels = []  # 记录每个病例的侵袭和分级标签

    elif args.dataset_name == 'derm7pt':
        count = 0
        correct = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        y_score = []
        y_true = []

        preds = []  # 记录每个病例的模型预测结果
        labels = []  # 记录每个病例的分类标签

    with torch.no_grad():
        for x, clinical, y, patient in dataloaders:
            count += 1
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            clinical = clinical.cuda()
            y = y.cuda()
            labels.append(y.cpu().numpy()[0])

            if args.method_name == 'CFCML':
                out = model(x, clinical, y, run_type='test')
            else:
                print('wrong method_name!')
            out_pred = out['out_prediction']

            print("\n-----------  " + str(count)+': '+ str(patient).split('\'')[1] + "  -----------")

            if args.dataset_name == 'men':
                out_pred = F.softmax(out_pred, dim=1)
                print(out_pred)
                pred = out_pred.argmax(dim=1, keepdim=False)
                all_preds.append(pred.cpu().numpy()[0])
                all_preds_score.append(out_pred.cpu().numpy()[0])
                print("predict: {}".format(pred.view_as(y)))
                print('target:  {}'.format(y))

            elif args.dataset_name == 'derm7pt':
                out_pred = F.softmax(out_pred, dim=1)
                pred = out_pred.argmax(dim=1, keepdim=True)
                y_score.append(out_pred[0,1].cpu().numpy())
                preds.append(pred.cpu().numpy()[0])
                y_true.append(y.cpu().numpy())

                print(out_pred)
                print("predict: {}".format(pred.view_as(y)))
                print('target:  {}'.format(y))

                correct += pred.eq(y.view_as(pred)).sum().item()
                if pred.view_as(y) == y == 1: TP += 1
                if (pred.view_as(y) == 1) and (y == 0): FP += 1
                if (pred.view_as(y) == 0) and (y == 1): FN += 1
                if pred.view_as(y) == y == 0: TN += 1

        if args.dataset_name =='men':
            right_0 = 0
            right_1 = 0
            right_2 = 0
            preds = np.array(all_preds)
            for i in range(preds.shape[0]):
                if preds[i] == labels[i] and preds[i] == 0:
                    right_0 += 1
                elif preds[i] == labels[i] and preds[i] == 1:
                    right_1 += 1
                elif preds[i] == labels[i] and preds[i] == 2:
                    right_2 += 1
            num_0 = len([i for i, x in enumerate(labels) if x == 0])
            num_1 = len([i for i, x in enumerate(labels) if x == 1])
            num_2 = len([i for i, x in enumerate(labels) if x == 2])

            acc_g1 = Decimal(right_0/num_0).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
            acc_g2inv = Decimal(right_1/num_1).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
            acc_g2noninv = Decimal(right_2/num_2).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
            acc = Decimal(accuracy_score(labels, all_preds)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
            weighted_f1 = Decimal(f1_score(labels, all_preds, average='weighted')).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
            macro_f1 = Decimal(f1_score(labels, all_preds, average='macro')).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
            auc = Decimal(roc_auc_score(one_hot_forauc(labels, num_classes), all_preds_score)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)

            print('\nacc_g1: {} ({}/{})'.format(acc_g1, right_0, num_0))
            print('acc_g2inv: {} ({}/{})'.format(acc_g2inv, right_1, num_1))
            print('acc_g2noinv: {} ({}/{})'.format(acc_g2noninv, right_2, num_2))
            print('acc:', acc)
            print('weighted_f1:', weighted_f1)
            print('macro_f1', macro_f1)
            print('auc', auc)

            result_ls = []
            result_ls.append(weight_date)

            result_ls.append(acc)
            result_ls.append(acc_g1)
            result_ls.append(acc_g2inv)
            result_ls.append(acc_g2noninv)
            result_ls.append(weighted_f1)
            result_ls.append(macro_f1)
            result_ls.append(auc)

            result_ls = np.reshape(result_ls, (1, len(result_ls)))
            if not os.path.exists(test_result_path):
                head_ls = ['weight_date', 'acc', 'acc_g1', 'acc_g2inv', 'acc_g2noninv', 'weighted_f1', 'macro_f1', 'auc']
                head_ls = np.reshape(head_ls, (1, len(head_ls)))
                result_out = np.concatenate((head_ls, result_ls), axis=0)
            else:
                result_in = pd.read_csv(test_result_path, header=None)
                result_out = np.concatenate((result_in, result_ls), axis=0)
            result_out_pd = pd.DataFrame(result_out)
            result_out_pd.to_csv(test_result_path, header=False, index=False)
            print("Finish saving result_men.csv!")

        elif args.dataset_name == 'derm7pt':
            fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
            roc_auc = AUC(fpr, tpr)
            auprc = average_precision_score(labels, preds)
            balance_acc = balanced_accuracy_score(labels, preds)
            mcc = matthews_corrcoef(labels, preds)

            print('\n')
            print('\nAcc_inv: {}/{} ({:.0f}%)'.format(correct, count, 100. * correct / count))
            print("TP,FP,FN,TN:", TP, FP, FN, TN)
            if not (TP == 0 and FP == 0):
                # print('precision:', TP_inv / (TP_inv + FP_inv))
                print('sensitivity:', TP / (TP + FN))
                print('specificity:', TN / (TN + FP))
                print('accuracy:', (TP + TN) / (TP + FP + FN + TN))
                print('g_mean:', ((TP / (TP + FN)) * (TN / (TN + FP))) ** 0.5)
            print('balance_acc:', str(balance_acc))
            print('mcc:', str(mcc))
            print('auprc:', str(auprc))
            print('roc_auc', roc_auc)

            result_ls = []
            result_ls.append(weight_date)

            result_ls.append(Decimal(TP / (TP + FN)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))
            result_ls.append(Decimal(TN / (TN + FP)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))
            result_ls.append(Decimal((TP + TN) / (TP + FP + FN + TN)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))
            result_ls.append(Decimal(((TP / (TP + FN)) * (TN / (TN + FP))) ** 0.5).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))
            result_ls.append(Decimal(balance_acc).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))
            result_ls.append(Decimal(auprc).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))
            result_ls.append(Decimal(roc_auc).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))

            result_ls = np.reshape(result_ls, (1, len(result_ls)))
            if not os.path.exists(test_result_path):
                head_ls = ['weight_date', 'Sen', 'Spe', 'Acc', 'G_mean', 'Ba_Acc', 'auprc', 'Auc']
                head_ls = np.reshape(head_ls, (1, len(head_ls)))
                result_out = np.concatenate((head_ls, result_ls), axis=0)
            else:
                result_in = pd.read_csv(test_result_path, header=None)
                result_out = np.concatenate((result_in, result_ls), axis=0)
            result_out_pd = pd.DataFrame(result_out)
            result_out_pd.to_csv(test_result_path, header=False, index=False)
            print("Finish saving to {}!".format(test_result_path))

            end_time = datetime.datetime.now()
            print('end_time: ' + str(end_time))


def find_saveinfsFortest():
    if args.dataset_name == 'men':
        if args.method_name == 'CFCML':
            save_infs = [['', '1fold'],
                         ['', '2fold'],
                         ['', '3fold']]

    elif args.dataset_name == 'derm7pt':
        if args.method_name == 'CFCML':
            save_infs = [[''],
                         [''],
                         ['']]
    return save_infs



if __name__ == '__main__':
    test_result_path = '../weights/result_{}.csv'.format(args.dataset_name)

    if args.dataset_name == 'men':
        data_path = ''
        clinical_path = ''
    elif args.dataset_name == 'derm7pt':
        data_path = ''
        clinical_path = ''

    print('start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if args.run_type == 'train':
        weight_path = False

        print('{} <<< epoch:{}, lr:{}, batch_size:{}, weight_decay:{}, cls_drop:{}'.format(args.run_type,
                  str(train_epochs), str(train_lr), str(batch_size), str(args.weight_decay_value), str(args.cls_drop)))

        if args.fold_train == 1:
            fold_list = ['1fold']
        elif args.fold_train == 2:
            fold_list = ['2fold']
        elif args.fold_train == 3:
            fold_list = ['3fold']
        elif args.fold_train == 12:
            fold_list = ['1fold','2fold']
        elif args.fold_train == 13:
            fold_list = ['1fold','3fold']
        elif args.fold_train == 23:
            fold_list = ['2fold','3fold']
        elif args.fold_train == 123:
            fold_list = ['1fold','2fold','3fold']
        else:
            print('wrong fold_train')

        if args.dataset_name == 'derm7pt':
            use_path = data_path
            train(batch_size, train_lr, weight_path)
        else:
            for fold in fold_list:
                print('processing------'+fold)
                use_path = os.path.join(data_path, fold)
                train(batch_size, train_lr, weight_path)

    elif args.run_type == 'test':  # test
        args.ifoffline_data_aug = 0
        save_infs = find_saveinfsFortest()
        for save_inf in save_infs:
            weight_date = save_inf[0]
            if args.dataset_name == 'derm7pt':
                use_path = data_path
                weight_saved_path = os.path.join('../weights', args.dataset_name, args.method_name, str(weight_date), 'train_weight_' + str(train_epochs) + '_' + str(batch_size) + '.pth')
            else:
                fold = save_inf[1]
                if args.fold_test == fold:
                    use_path = os.path.join(data_path, fold)
                    weight_saved_path = os.path.join('../weights', args.dataset_name, args.method_name, str(weight_date), fold, fold + '_train_weight_' + str(train_epochs) + '_' + str(batch_size) + '.pth')

            test(weight_saved_path, num_classes, weight_date)

    else:
        print('Wrong run_type!')
    print('\nend time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


