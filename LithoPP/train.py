import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader

from data import Dataset
from model import EdgeNet


if __name__ == "__main__":
    train_dataset = Dataset(is_val=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=40, # load all in one time
        shuffle=True
    )

    test_dataset = Dataset(is_val=True)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=10 # load all in one time
    )

    device = torch.device("mps:0" if torch.mps.is_available() else "cpu")
    model = EdgeNet().to(device)

    # 训练
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    best_val_loss = np.inf

    for i, epoch in enumerate(range(200)):
        model.train()
        train_loss, train_iter = 0, 0
        for x, y in train_dataloader:
            pred = model(x)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_iter += 1

        model.eval()
        with torch.no_grad():
            for x, y in test_dataloader:
                val_y = model(x)
                val_loss = criterion(val_y, y)

        print(f"Epoch {i}, Train Loss: {train_loss}, Val Loss: {val_loss.item()}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "../groundtruth/best_edge_model.pth")  # 保存最佳模型
            print("------ Saving model")

    print("训练完成")

    model.load_state_dict(torch.load("../groundtruth/best_edge_model.pth", weights_only=True))
    model.eval()
    with torch.no_grad():
        for x, y in test_dataloader:
            val_y = model(x)
            val_loss = criterion(val_y, y)
        input = x.cpu().numpy()
        predictions = val_y.cpu().numpy()
        ground_truths = y.cpu().numpy()

    print("\n=== 最终验证结果 ===")
    for i in range(min(10, len(predictions))):
        x = input[i]
        pred = predictions[i] * 30
        gt = ground_truths[i] * 30
        print(f"样本 {i + 1}: {x}")
        print(f"  真实值: 内边缘: {int(gt[0])}, 中心点: {int(gt[1])}, 外边缘: {int(gt[2])}")
        print(f"  预测值: 内边缘: {int(pred[0])}, 中心点: {int(pred[1])},  外边缘: {int(pred[2])}\n")

    print("验证评估完成")
