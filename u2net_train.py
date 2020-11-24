import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import (AlbuSampleTransformer, MixupAugSalObjDataset,
                         MultiScaleSalObjDataset, RandomCrop, RescaleT, SaveDebugSamples,
                         SalObjDataset, ToTensorLab, get_heavy_transform)
from model import U2NET, U2NETP, CustomNet
import albumentations as A
import cv2

bce_loss = nn.BCELoss(size_average=True)


def multi_scale_collater(batch):
    sizes = list(range(256, 640+1, 32))
    scale = (0.5, 1.5)
    ratio = (0.5, 2.0)
    size = np.random.choice(sizes)

    _tr = transforms.Compose([
        AlbuSampleTransformer(A.RandomResizedCrop(width=size, height=size,
                                                scale=scale,
                                                ratio=ratio,
                                                interpolation=cv2.INTER_LINEAR)),
        ToTensorLab(flag=0)
    ])

    transformed_batch = []
    for sample in batch:
        assert "image" in sample
        assert "label" in sample
        assert "imidx" in sample

        transformed_batch.append(_tr(sample))

    data = {}
    for key in batch[0].keys():
        if key == "imidx":
            data[key] = list(map(lambda s: s[key], transformed_batch))
        else:
            data[key] = torch.stack(list(map(lambda s: s[key], transformed_batch)))

    return data


def multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


def multi_bce_loss_fusion5(d0, d1, d2, d3, d4, d5, labels_v):

    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5

    return loss0, loss


def main():
    # ---------------------------------------------------------
    # Configurations
    # ---------------------------------------------------------

    heavy_augmentation = True  # False to use author's default implementation
    # mutually exclusive with multiscale_training
    mixup_augmentation = False
    # mutually exclusive with mixup_augmentation
    multiscale_training = True
    multi_gpu = True
    mixed_precision_training = True

    model_name = 'u2net'  # 'u2netp'
    se_type = None   # "csse", "sse", "cse", None; None to use author's default implementation
    checkpoint = "saved_models/u2net/u2net_mixed_person_n_portraits_heavy_aug_multiscale_bce_itr_18000_train_0.331498_tar_0.041404.pth"

    # data_dir = '../datasets/'
    # tra_image_dir = '123rf_person_removebg/image/'
    # tra_label_dir = '123rf_person_removebg/alpha/'

    train_dirs = [
        # "../data/open_images_person_6k/",
        "../data/123rf_person_removebg/",
        "../data/supervisely_person/",
        "../data/portraits/",
        "../data/aisegment_portraits/",

        "../data/open_images_v6_person_remove_bg_chunk_0_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_1_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_2_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_3_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_4_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_5_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_6_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_7_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_8_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_9_paired/",
        "../data/open_images_v6_person_remove_bg_chunk_10_paired/",
    ]
    train_dirs_file_limit = [
        # None,
        None,
        None,
        None,
        30000,

        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]

    image_ext = '.jpg'
    label_ext = '.png'
    dataset_name = "mixed_person_n_portraits"

    lr = 0.0003
    epoch_num = 300
    batch_size_train = 8
    batch_size_val = 1
    workers = 24
    save_frq = 2000  # save the model every 2000 iterations

    save_debug_samples = True
    debug_samples_dir = "./debug/"

    # ---------------------------------------------------------

    model_dir = './saved_models/' + model_name + '/'
    os.makedirs(model_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Construct data input pipeline
    # ---------------------------------------------------------

    # Get dataset name
    # if not dataset_name:
    #     dataset_name = tra_image_dir.split(sep=os.path.sep)[0]
    dataset_name = dataset_name.replace(" ", "_")

    # Get training data
    assert len(train_dirs) == len(train_dirs_file_limit), "Different train dirs and train dirs file limit length!"

    tra_img_name_list = []
    tra_lbl_name_list = []
    for d, flimit in zip(train_dirs, train_dirs_file_limit):
        img_files = glob.glob(d + '**/*' + image_ext, recursive=True)
        if flimit:
            img_files = np.random.choice(img_files, size=flimit, replace=False)

        print(f"directory: {d}, files: {len(img_files)}")

        for img_path in img_files:
            lbl_path = img_path.replace("/image/", "/alpha/") \
                .replace(image_ext, label_ext)

            if os.path.exists(img_path) and os.path.exists(lbl_path):
                tra_img_name_list.append(img_path)
                tra_lbl_name_list.append(lbl_path)

    train_num = len(tra_img_name_list)
    # val_num = 0  # unused
    print(f"dataset name        : {dataset_name}")
    print(f"training samples    : {train_num}")

    # Construct data input pipeline
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

    # Create dataset and dataloader
    dataset_kwargs = dict(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose(
            [transform, ]
            + ([SaveDebugSamples(out_dir=debug_samples_dir), ] if save_debug_samples else [])
            + ([ToTensorLab(flag=0), ] if not multiscale_training else [])
        )
    )
    if mixup_augmentation:
        _dataset_cls = MixupAugSalObjDataset
    # elif multiscale_training:
    #     _dataset_cls = MultiScaleSalObjDataset
    else:
        _dataset_cls = SalObjDataset

    salobj_dataset = _dataset_cls(**dataset_kwargs)
    salobj_dataloader = DataLoader(salobj_dataset,
                                   batch_size=batch_size_train,
                                   collate_fn=multi_scale_collater if multiscale_training else None,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=workers)

    # ---------------------------------------------------------
    # 2. Load model
    # ---------------------------------------------------------

    # Instantiate model
    if model_name == 'u2net':
        net = U2NET(3, 1, se_type=se_type)
    elif model_name == 'u2netp':
        net = U2NETP(3, 1, se_type=se_type)
    elif model_name == 'custom':
        net = CustomNet()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Restore model weights from checkpoint
    if checkpoint:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

        print(f"Restoring from checkpoint: {checkpoint}")
        try:
            net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            print(" - [x] success")
        except:
            print(" - [!] error")

    if torch.cuda.is_available():
        net.cuda()

    # ---------------------------------------------------------
    # 3. Define optimizer
    # ---------------------------------------------------------

    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0)

    # ---------------------------------------------------------
    # 4. Initialize AMP and data parallel stuffs
    # ---------------------------------------------------------

    GOT_AMP = False
    if mixed_precision_training:
        try:
            print("Checking for Apex AMP support...")
            from apex import amp
            GOT_AMP = True
            print(" - [x] yes")
        except ImportError:
            print(" - [!] no")

    if GOT_AMP:
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(net, optimizer,
                                          opt_level="O1")

    if torch.cuda.device_count() > 1 and multi_gpu:
        print(f"Multi-GPU training using {torch.cuda.device_count()} GPUs.")
        net = nn.DataParallel(net)
    else:
        print(f"Training using {torch.cuda.device_count()} GPUs.")

    # ---------------------------------------------------------
    # 5. Training
    # ---------------------------------------------------------

    print("Start training...")

    ite_num = 0
    ite_num4val = 0
    running_loss = 0.0
    running_tar_loss = 0.0

    for epoch in tqdm(range(0, epoch_num), desc="Epoch"):
        net.train()

        for i, data in enumerate(tqdm(salobj_dataloader, desc="Batch")):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            image_key = "image"
            label_key = "label"

            # # Randomly select one of available sizes if multiscale training
            # if multiscale_training:
            #     size = np.random.choice(salobj_dataloader.dataset.sizes)
            #     tqdm.write(f"current input size: {size}")

            #     image_key = f"image_{size}"
            #     label_key = f"label_{size}"

            inputs, labels = data[image_key], data[label_key]
            tqdm.write(f"input tensor shape: {inputs.shape}")

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # Wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = \
                    Variable(inputs.cuda(), requires_grad=False), \
                    Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = \
                    Variable(inputs, requires_grad=False), \
                    Variable(labels, requires_grad=False)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            d6 = 0
            if model_name == "custom":
                d0, d1, d2, d3, d4, d5 = net(inputs_v)
                loss2, loss = multi_bce_loss_fusion5(d0,
                                                     d1,
                                                     d2,
                                                     d3,
                                                     d4,
                                                     d5,
                                                     labels_v)
            else:
                d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
                loss2, loss = multi_bce_loss_fusion(d0,
                                                    d1,
                                                    d2,
                                                    d3,
                                                    d4,
                                                    d5,
                                                    d6,
                                                    labels_v)

            if GOT_AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # Delete temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            # Print stats
            tqdm.write("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num,
                (i + 1) * batch_size_train, train_num,
                ite_num,
                running_loss / ite_num4val,
                running_tar_loss / ite_num4val
            ))

            if ite_num % save_frq == 0:
                # Save checkpoint
                torch.save(net.module.state_dict() if hasattr(net, "module") else net.state_dict(),
                           model_dir
                           + model_name
                           + (("_" + se_type) if se_type else "")
                           + ("_" + dataset_name)
                           + ("_mixup_aug" if mixup_augmentation else "")
                           + ("_heavy_aug" if heavy_augmentation else "")
                           + ("_multiscale" if multiscale_training else "")
                           + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
                    ite_num,
                    running_loss / ite_num4val,
                    running_tar_loss / ite_num4val
                ))

                # Reset stats
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

    print("Training completed successfully.")


if __name__ == "__main__":
    main()
