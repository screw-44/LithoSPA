import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import FinetuneDataset


from segment_anything import sam_model_registry
from tqdm import tqdm
import monai
import torch.nn as nn
import torch.optim as optim

# 参考 https://github.com/mazurowski-lab/finetune-SAM/blob/main/SingleGPU_train_finetune_noprompt.py

if __name__ == "__main__":
    dataset = FinetuneDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    device = "cuda:0"
    sam_checkpoint = "./checkpoint/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.train()
    sam.to(device=device)

    num_epochs = 5
    optimizer = optim.AdamW(sam.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True, reduction='mean')
    criterion2 = nn.CrossEntropyLoss()

    for i, epoch in enumerate(range(num_epochs)):
        epoch_losses = []
        train_loss, iter_num = 0, 0
        for batch in tqdm(dataloader):
            imgs = batch["adi"].to(device)
            msks = batch["label"].to(device)
            img_emb = sam.image_encoder(imgs)

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

            pred, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
            loss_dice = criterion1(pred, msks.float())
            loss_ce = criterion2(pred, torch.squeeze(msks.long(), 1))
            loss = loss_dice + loss_ce

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            iter_num += 1

        print("loss:", loss)

        torch.save(sam.state_dict(), f"./experiment/exp3/epoch{i}.pth")
