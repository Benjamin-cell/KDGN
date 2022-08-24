import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os

from collections import defaultdict

from models import KDGN
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params, buildMPNN

torch.manual_seed(1203)
np.random.seed(1203)

model_name = 'KDGN'
resume_name = ''

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_name, help='resume path')
parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")

args = parser.parse_args()
model_name = args.model_name
resume_name = args.resume_path

### 2022-7-14 hill loss
# class Hill(nn.Module):
#     r""" Hill as described in the paper "Robust Loss Design for Multi-Label Learning with Missing Labels "
#
#     .. math::
#         Loss = y \times (1-p_{m})^\gamma\log(p_{m}) + (1-y) \times -(\lambda-p){p}^2
#
#     where : math:`\lambda-p` is the weighting term to down-weight the loss for possibly false negatives, 是加权项，用于降低可能的假阴性损失的权重
#           : math:`m` is a margin parameter,
#           : math:`\gamma` is a commonly used value same as Focal loss. 是和 Focal loss 一样的常用值。
#
#     .. note::
#         Sigmoid will be done in loss.
#
#     Args:
#         lambda (float): Specifies the down-weight term. Default: 1.5. (We did not change the value of lambda in our experiment.指定权重项。 默认值：1.5。 （我们在实验中没有改变 lambda 的值。）)
#         margin (float): Margin value. Default: 1 . (Margin value is recommended in [0.5,1.0], and different margins have little effect on the result. margin值建议在[0.5,1.0]，不同的margin对结果影响不大。）)
#         gamma (float): Commonly used value same as Focal loss. Default: 2
#
#     """
#
#     def __init__(self, lamb: float = 1.5, margin: float = 1.0, gamma: float = 2.0,  reduction: str = 'sum') -> None:
#         super(Hill, self).__init__()
#         self.lamb = lamb
#         self.margin = margin
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, logits, targets):
#         """
#         call function as forward
#
#         Args:
#             logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
#             targets : Multi-label binarized vector with shape of :math:`(N, C)`
#
#         Returns:
#             torch.Tensor: loss
#         """
#
#         # Calculating predicted probability
#         logits_margin = logits - self.margin
#         pred_pos = torch.sigmoid(logits_margin)
#         pred_neg = torch.sigmoid(logits)
#
#         # Focal margin for postive loss
#         pt = (1 - pred_pos) * targets + (1 - targets)
#         focal_weight = pt ** self.gamma
#
#         # Hill loss calculation
#         los_pos = targets * torch.log(pred_pos)
#         los_neg = (1-targets) * -(self.lamb - pred_neg) * pred_neg ** 2
#
#
#         loss = -(los_pos + los_neg) ## -(lambda -p)
#         loss *= focal_weight ## p2
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss
# miss_loss = Hill()


class ACF(nn.Module):
    ##

    # .. note::
    #     ACF can be combinded with various multi-label loss functions.
    #     ACF performs best combined with Focal margin loss in our paper. Code of acf with Focal margin loss is released here.
    #     Since the first epoch can recall few missing labels with high precision, acf can be used ater the first epoch.
    #     Sigmoid will be done in loss.
    #
    # Args:
    #     tau (float): threshold value. Default: 0.6
    #     change_epoch (int): which epoch to combine acf. Default: 1
    #     margin (float): Margin value. Default: 1
    #     gamma (float): Hard mining value. Default: 2
    #     reduction (string, optional): Specifies the reduction to apply to the output:
    #         ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    #         ``'mean'``: the sum of the output will be divided by the number of
    #         elements in the output, ``'sum'``: the output will be summed. Default: ``'sum'``
    #
    #     """

    def __init__(self,
                 tau: float = 0.8,
                 change_epoch: int = 1,
                 margin: float = 1.0,
                 gamma: float = 3.0,
                 reduction: str = 'sum') -> None:
        super(ACF, self).__init__()
        self.tau = tau
        self.change_epoch = change_epoch
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.LongTensor,
                epoch) -> torch.Tensor:
        """
        call function as forward

        Args:
            logits : The predicted logits before sigmoid with shape of :math:`(N, C)`
            targets : Multi-label binarized vector with shape of :math:`(N, C)`
            epoch : The epoch of current training.

        Returns:
            torch.Tensor: loss
        """
        # Subtract margin for positive logits
        logits = torch.where( targets == 1, logits - self.margin, logits )

        # ACF missing label correction
        if epoch >= self.change_epoch:

            targets = torch.where(torch.sigmoid(logits) > self.tau, torch.tensor(1).cuda(), targets)

        pred = torch.sigmoid(logits)

        # Focal margin for postive loss
        pt = (1 - pred) * targets + pred * (1 - targets)
        focal_weight = pt ** self.gamma

        los_pos = targets * F.logsigmoid(logits)
        los_neg = (1 - targets) * F.logsigmoid(-logits)

        loss = -(los_pos + los_neg)
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
function = ACF()
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss




def eval(model, data_eval, voc_size, epoch):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0

    for step, input in enumerate(data_eval):
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        for adm_idx, adm in enumerate(input):


            target_output1 = model(input[:adm_idx+1])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)


            target_output1 = F.sigmoid(target_output1).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp>=0.5] = 1
            y_pred_tmp[y_pred_tmp<0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)



        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}

        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    llprint('\tDDI Rate: %.4f, Jaccard: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f\n' % (
        ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)
    ))
    dill.dump(obj=smm_record, file=open('../data/KDGN_records.pkl', 'wb'))
    dill.dump(case_study, open(os.path.join('saved', model_name, 'case_study.pkl'), 'wb'))

    # print('avg med', med_cnt / visit_cnt)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

    # def poly1_b_cross_entropy_torch( logits, labels, class_number=112, epsilon=1.0):
    #     poly1 = torch.sum(F.one_hot(labels.to(torch.int64), class_number).float() * F.softmax(logits), dim=-1)
    #
    #     loss_bce = F.binary_cross_entropy_with_logits(torch.FloatTensor(logits), torch.FloatTensor(labels))
    #     poly1_ce_loss = (loss_bce + epsilon * (1 - poly1))
    #     return poly1_ce_loss


    ddi_mask_path = '../data/ddi_mask_H.pkl'
    molecule_path = '../data/atc3toSMILES.pkl'
    ddi_mask_H = dill.load(open(ddi_mask_path, 'rb'))
    molecule = dill.load(open(molecule_path, 'rb'))



    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'
    ehr_adj_path = '../data/ehr_adj_final_newone.pkl'
    ddi_adj_path = '../data/ddi_A_final_new.pkl'
    device = torch.device('cuda')

    ehr_adj = dill.load(open(ehr_adj_path, 'rb'))
    ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]

    EPOCH = 30
    LR = 0.0002
    TEST = args.eval
    Neg_Loss = args.ddi
    DDI_IN_MEM = args.ddi
    TARGET_DDI = 0.05
    T = 0.5
    decay_weight = 0.85

    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    MPNNSet, N_fingerprint, average_projection = buildMPNN(molecule, med_voc.idx2word, 2, device)

    model = KDGN(voc_size, ehr_adj, ddi_adj, ddi_mask_H, MPNNSet, N_fingerprint, average_projection, emb_dim=64,
                    device=device, ddi_in_memory=DDI_IN_MEM)


    if TEST:
        model.load_state_dict(torch.load(open(resume_name, 'rb')))
    model.to(device=device)

    print('parameters', get_n_params(model))
    optimizer = Adam(list(model.parameters()), lr=LR)

    if TEST:
        eval(model, data_test, voc_size, 0)
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        for epoch in range(EPOCH):
            loss_record1 = []
            start_time = time.time()
            model.train()
            prediction_loss_cnt = 0
            neg_loss_cnt = 0
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):


                    seq_input = input[:idx+1]

                    loss1_target = np.zeros((1, voc_size[2]))
                    
                    loss1_target[:, adm[2]] = 1
                    loss3_target = np.full((1, voc_size[2]), -1)
                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx] = item

                    target_output1 = model(seq_input)





                    loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss1_target).to(device))
                    loss3 = F.multilabel_margin_loss(F.sigmoid(target_output1), torch.LongTensor(loss3_target).to(device))



                    ### 2022-5-12 
                    # loss_bce = F.binary_cross_entropy(target_output1, torch.FloatTensor(loss1_target).to(device))
                    # loss0 = poly1_b_cross_entropy_torch( target_output1, torch.FloatTensor(loss1_target))
                    # print(loss0,'*************')
                    # loss5 = multilabel_categorical_crossentropy(torch.FloatTensor(loss1_target).to(device),target_output1)

                    loss2 = function(target_output1,torch.FloatTensor(loss1_target).to(device))

                    loss = loss2 + loss1 / (loss1 / loss2).detach() + loss3 / (loss3 / loss2).detach()

                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    loss_record1.append(loss.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, L_p cnt: %d, L_neg cnt: %d' % (epoch, step, len(data_train), prediction_loss_cnt, neg_loss_cnt))
            # annealing
            T *= decay_weight

            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch)

            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            llprint('\tEpoch: %d, Loss: %.4f, One Epoch Time: %.2fm, Appro Left Time: %.2fh\n' % (epoch,
                                                                                                np.mean(loss_record1),
                                                                                                elapsed_time,
                                                                                                elapsed_time * (
                                                                                                            EPOCH - epoch - 1)/60))

            torch.save(model.state_dict(), open( os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
            print('')
            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja


        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))

        # test
        torch.save(model.state_dict(), open(
            os.path.join('saved', model_name, 'final.model'), 'wb'))

        print('best_epoch:', best_epoch)


if __name__ == '__main__':
    main()
