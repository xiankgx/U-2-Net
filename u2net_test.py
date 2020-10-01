import glob
import os

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_loader import RescaleT, SalObjDataset, ToTensorLab
from model import U2NET, U2NETP, CustomNet


def normPRED(d):
    # normalize the predicted SOD probability map
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi)/(ma - mi)
    return dn


def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    # imo.save(d_dir+imidx+'.png')
    # print(f"image.shape: {image.shape}")
    # print(f"pb_np.shape: {pb_np.shape}")
    #io.imsave(d_dir+imidx+'.png', np.concatenate([image, pb_np[..., [0]]], -1))
    io.imsave(d_dir + imidx + '.png', pb_np[..., 0])


def main():

    full_sized_testing = False
    model_name = 'custom'
    # model_name = 'u2netp'

    image_dir = '/home/gx/datasets/DUTS-TE/DUTS-TE-Image/'
    # prediction_dir = './test_data/' + model_name + '_results/'
    # model_dir = './saved_models/' + model_name + '/' + model_name + '.pth'

    model_dir = "./saved_models/custom/custom_heavy_aug_bce_itr_248000_train_0.396976_tar_0.067510.pth"
    prediction_dir = f"./predictions{'_fullsize' if full_sized_testing else ''}_{os.path.splitext(os.path.basename(model_dir))[0]}/"

    os.makedirs(prediction_dir, exist_ok=True)

    img_name_list = glob.glob(image_dir + '*')

    img_exts = [".jpg", ".jpeg", ".png", ".jfif"]
    img_name_list = list(filter(lambda p: os.path.splitext(p)[-1].lower() in img_exts,
                                img_name_list))
    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose(
                                            ([] if full_sized_testing else [
                                             RescaleT(320), ])
                                            + [ToTensorLab(flag=0), ]
                                        ))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif(model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    elif(model_name == 'custom'):
        net = CustomNet()
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d7 = 0
        if model_name == "custom":
            d1, d2, d3, d4, d5, d6 = net(inputs_test)
        else:
            d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    main()
