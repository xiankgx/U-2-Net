import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loader import (AlbuSampleTransformer, MixupAugSalObjDataset,
                         MultiScaleSalObjDataset, RandomCrop, Rescale,
                         RescaleT, SalObjDataset, ToTensor, ToTensorLab,
                         get_heavy_transform)
from model import U2NET, U2NETP

bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    # print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    #     loss0.item(),
    #     loss1.item(),
    #     loss2.item(),
    #     loss3.item(),
    #     loss4.item(),
    #     loss5.item(),
    #     loss6.item()
    # ))

    return loss0, loss


def main():
    # ---------------------------------------------------------
    # Configs
    # ---------------------------------------------------------

    checkpoint = "./saved_models/u2net/u2net_heavy_aug__bce_itr_4000_train_1.834341_tar_0.250207.pth"
    mixup_augmentation = False
    heavy_augmentation = True
    multiscale_training = False
    multi_gpu = False

    model_name = 'u2net'  # 'u2netp'

    data_dir = '../data/'
    tra_image_dir = 'DUTS-TR/DUTS-TR-Image/'
    tra_label_dir = 'DUTS-TR/DUTS-TR-Mask/'
    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = './saved_models/' + model_name + '/'
    os.makedirs(model_dir, exist_ok=True)

    lr = 0.001
    epoch_num = 195
    batch_size_train = 12
    batch_size_val = 1
    workers = 16
    save_frq = 2000  # save the model every 2000 iterations

    # ---------------------------------------------------------

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split("/")[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)
    val_num = 0

    if heavy_augmentation:
        transform = AlbuSampleTransformer(
            get_heavy_transform(
                transform_size=False if multiscale_training else True)
        )
    else:
        transform = transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
        ])

    dataset_kwargs = dict(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose(
            [transform, ]
            + ([ToTensorLab(flag=0), ] if not multiscale_training else [])
        )
    )
    if mixup_augmentation:
        _dataset_cls = MixupAugSalObjDataset
    elif multiscale_training:
        _dataset_cls = MultiScaleSalObjDataset
    else:
        _dataset_cls = SalObjDataset

    salobj_dataset = _dataset_cls(**dataset_kwargs)
    salobj_dataloader = DataLoader(salobj_dataset,
                                   batch_size=batch_size_train,
                                   shuffle=True,
                                   num_workers=workers)

    # ------- 3. define model --------
    # define the net
    if (model_name == 'u2net'):
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        net = U2NETP(3, 1)

    if checkpoint:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

        print(f"Restoring from checkpoint: {checkpoint}")
        try:
            net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            print("-- success")
        except:
            print("-- error")

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0)

    if torch.cuda.device_count() > 1 and multi_gpu:
        print(f"Multi-GPU training using {torch.cuda.device_count()} GPUs.")
        net = nn.DataParallel(net)
    else:
        print(f"Training using {torch.cuda.device_count()} GPUs.")

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    ite_num4val = 0
    running_loss = 0.0
    running_tar_loss = 0.0

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            image_key = "image"
            label_key = "label"
            if multiscale_training:
                size = np.random.choice(salobj_dataloader.dataset.sizes)
                # print(f"size: {size}")
                image_key = f"image_{size}"
                label_key = f"label_{size}"

            inputs, labels = data[image_key], data[label_key]
            # print(f"{inputs.shape}")

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = \
                    Variable(inputs.cuda(), requires_grad=False), \
                    Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = \
                    Variable(inputs, requires_grad=False), \
                    Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0,
                                               d1,
                                               d2,
                                               d3,
                                               d4,
                                               d5,
                                               d6,
                                               labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num,
                (i + 1) * batch_size_train, train_num,
                ite_num,
                running_loss / ite_num4val,
                running_tar_loss / ite_num4val
            ))

            if ite_num % save_frq == 0:

                torch.save(net.module.state_dict() if hasattr(net, "module") else net.state_dict(),
                           model_dir
                           + model_name
                           + ("_mixup_aug_" if mixup_augmentation else "")
                           + ("_heavy_aug_" if heavy_augmentation else "")
                           + ("_multiscale_" if multiscale_training else "")
                           + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                    ite_num,
                    running_loss / ite_num4val,
                    running_tar_loss / ite_num4val
                ))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0


if __name__ == "__main__":
    main()
