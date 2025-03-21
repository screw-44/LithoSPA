import argparse
import torch
import monai
import torchvision

import torch.optim as optim
import os

from tqdm import trange, tqdm
from torch import nn

from models.unet import Unet
from models.vit import get_b16_config
from models.vit import VisionTransformer as ViT_seg
from data.data_loader import AlignedDatasetLoader



def get_model(_model):
    if _model == "unet":
        return Unet()
    elif _model == "vit":
        config = get_b16_config()
        config.n_classes, config.n_skip = 1, 0
        return ViT_seg(config, img_size=256, num_classes=1)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="vit", )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--layout_image_dir", default="../../sjw/layout2adi0907/train/layout")
    parser.add_argument("--sem_image_dir", default="../../sjw/layout2adi0907/train/ADI")
    parser.add_argument("--gt_dir", default="../dataset/seg_4_epoch3")
    parser.add_argument("--device", default="mps:0")
    parser.add_argument("--max_data_size", type=int, default=50)
    args = parser.parse_args()


    data_loader = AlignedDatasetLoader(args)
    dataset = data_loader.load_data()
    print("# training dataset size:{}".format(len(data_loader)))

    model = get_model(args.model).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True, reduction='mean')
    criterion2 = nn.CrossEntropyLoss()

    for epoch in trange(0, 100):
        for data in tqdm(dataset):
            sem, gt = data['sem'].to(args.device), data['gt'].to(args.device)
            pred = model.forward(sem)

            loss = criterion1(pred, gt) + criterion2(pred, torch.squeeze(gt.long(), 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        print('Epoch: {} has loss: {}'.format(epoch, loss))
        if epoch % 1 == 0:
            print('saving the model and image at epoch: {}'.format(epoch))
            show_image = torch.cat((sem,  gt, pred), dim=0)
            grid = torchvision.utils.make_grid(show_image, args.batch_size)
            torchvision.utils.save_image(grid, os.path.join("./experiment/", 'epoch_' + str(epoch) + '.png'))

            # save_network(model, experiment_path, model.name, epoch)
            # torchvision.utils.save_image(layout, os.path.join(experiment_path, 'epoch_' + str(epoch) + '_layout.png'))