import cv2
from torch.utils.data import DataLoader
from dataset import FinetuneDataset
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm, trange
import monai
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


from utils import visualize_tensor

if __name__ == "__main__":
    dataset = FinetuneDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=1, # 只能设为1，因为pytorch不支持同一个batch中dataloader加载不同大小的数据
        shuffle=True
    )

    num_in_epochs = 10 # 每一轮自监督，内部训练4个epoch
    num_out_epochs = 10
    # 外轮训练10个epoch，外轮每次都从base开始。不分内外轮的实现，可能会导致模型灾难性遗忘，从有内外轮实现开始。
    # 考虑到实现难度，我们先实现直接None去接近Prompt。如果效果不行，就实现None(三轮），再接近Prompt。
    device = "cuda:3"
    sam_checkpoint = "./checkpoint/sam_vit_b_01ec64.pth"
    # sam_checkpoint = "experiment/exp1/d1_epoch3.pth"
    #3/1 排除了代码问题，目前问题是prompt encoder也会跟着训练
    # 将prompt encoder固定后，模型不会让prompt和label去接近。但是，在3个epoch后，出现了灾难性遗忘的问题，分割性能显著下降。而1epoch的时候，性能是有改善的。
    model_type = "vit_b"
    # 没有prompt的输入要接近 带box prompt的输入
    for out_loop in range(num_out_epochs):

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.train()
        for n, value in sam.prompt_encoder.named_parameters():
            value.requires_grad = False

        sam.to(device=device)

        optimizer = optim.AdamW(sam.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1,
                                amsgrad=False)
        criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True, reduction='mean')
        criterion2 = nn.CrossEntropyLoss()
        sam_transform = ResizeLongestSide(sam.image_encoder.img_size)

        for in_loop in range(num_in_epochs):
            epoch_losses = []
            for batch in tqdm(dataloader):
                adi_image = batch["adi"].to(device)
                adi_img = sam.preprocess(adi_image * 255) # fucking sam use 255 value as input
                img_emb = sam.image_encoder(adi_img)

                box_prompt = batch["box_prompt"].to(device)
                transform_boxes = sam_transform.apply_boxes_torch(box_prompt, (1024, 1024))

                # box prompt的输出
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=None,
                    boxes=transform_boxes,
                    masks=None,
                )
                # bbox的输出应该和box的输入量是一样的
                pred_value_bbox, _ = sam.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                # 相同大小， 不用resize pred_bbox = sam.postprocess_masks(pred_value_bbox, (256, 256), (256, 256))
                pred_bbox = pred_value_bbox > sam.mask_threshold
                pred_bbox = torch.any(pred_bbox, dim=0)
                pred_bbox = torch.unsqueeze(pred_bbox, 0).float()
                # print(pred_bbox_final_result)
                # 可视化预测结果，训练过程中查看训练进度
                # visualize_tensor(pred_bbox)
                # visualize_tensor(adi_image)
                # visual_img = adi_image.squeeze(0).permute(1,2, 0).cpu().numpy()*255.0
                # visual_img = visual_img.astype(np.uint8)
                # visual_img = np.ascontiguousarray(visual_img)
                # for box in box_prompt[0]:
                #     cv2.rectangle(visual_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 4)
                # cv2.imshow("visual img", visual_img)
                # cv2.waitKey(0)

                # None情况下的输出
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )
                pred_none, _ = sam.mask_decoder(
                    image_embeddings=img_emb,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=True,
                )
                # visualize_tensor(pred_none)

                # 计算两者的loss, 让none去接近bbox
                loss_dice = criterion1(pred_none, pred_bbox)
                loss_ce = criterion2(pred_none, torch.squeeze(pred_bbox.long(), 1))
                loss = loss_dice + loss_ce

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            visualize_tensor(pred_bbox)
            visualize_tensor(adi_image)
            visualize_tensor(pred_none)
            print(f"loss:{loss}, image_id:"+batch["index_dir"][0])
            torch.save(sam.state_dict(), f"./experiment/ss_pure_contrast_fbase_exp1/epoch{in_loop}.pth")
        break


