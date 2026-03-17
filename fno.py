import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import torch.nn.functional as F
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--max_steps', type = int, default =5000)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--lr_sch', type = str, default = 'onecycle')
parser.add_argument('--modes1', type = int, default = 64)  # 傅里叶模态数
parser.add_argument('--modes2', type = int, default = 64)  # 傅里叶模态数
parser.add_argument('--width', type = int, default = 32)  # 网络宽度
args = parser.parse_args()


# 数据加载器（保持不变）
class PDEDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        input_grid, target_grid = torch.load(self.data_paths[idx])
        input_tensor = input_grid.permute(2, 0, 1).float()
        target_tensor = target_grid.permute(2, 0, 1).float()
        return input_tensor, target_tensor


# FNO模型定义
class FNO2d(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, modes1 = 32, modes2 = 32, width = 64):
        super(FNO2d, self).__init__()
        """
        参数说明：
        in_channels: 输入通道数 (x, t, λ)
        out_channels: 输出通道数 (c)
        modes1, modes2: 傅里叶模态数（空间维度）
        width: 网络宽度
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # 输入投影（升维）
        self.p = nn.Conv2d(in_channels, self.width, 1)

        # 傅里叶层
        self.fourier_layers = nn.ModuleList([
            FourierLayer(self.width, self.width, self.modes1, self.modes2)
            for _ in range(4)
        ])

        # 局部卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(4)
        ])

        # 输出投影
        self.q = nn.Conv2d(self.width, out_channels, 1)

    def forward(self, x):
        # 输入形状: [batch, 3, H, W]
        x = self.p(x)  # 升维到 [batch, width, H, W]

        # 傅里叶层处理
        for (f_layer, c_layer) in zip(self.fourier_layers, self.conv_layers):
            x_f = f_layer(x)    #频域操作
            x_c = c_layer(x)#空域操作
            x = x_f + x_c  # 残差连接
            x = F.gelu(x)

        # 输出投影
        x = self.q(x)  # [batch, 1, H, W]
        return x


# class FourierLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(FourierLayer, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1  # 傅里叶模态数（空间维度1）
#         self.modes2 = modes2  # 傅里叶模态数（空间维度2）
#
#         # 傅里叶权重参数（复数用两个实数表示）
#         self.scale = 1 / (in_channels * out_channels)
#         self.weights1 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
#         self.weights2 = nn.Parameter(
#             self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2))
#
#     def compl_mul2d(self, input, weights):
#         # 复数乘法操作 (batch, in_channel, x,y), (in_channel, out_channel, x,y)
#         return torch.einsum("bixy,ioxy->boxy", input, weights)
#
#     def forward(self, x):
#         batchsize = x.shape[0]
#         H, W = x.shape[-2], x.shape[-1]
#
#         # 傅里叶变换
#         x_ft = torch.fft.rfft2(x)
#
#         # 频域滤波
#         out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1,
#                              device = x.device, dtype = torch.cfloat)
#
#         # 处理低频模态
#         modes1 = min(self.modes1, H)
#         modes2 = min(self.modes2, W // 2 + 1)
#
#         # 维度1的低频部分
#         out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(x_ft[:, :, :modes1, :modes2],
#             torch.view_as_complex(self.weights1[:, :, :modes1, :modes2]))
#
#         # 维度2的低频部分
#         out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(x_ft[:, :, -modes1:, :modes2],
#             torch.view_as_complex(self.weights2[:, :, -modes1:, :modes2]))
#
#         # 逆傅里叶变换
#         x = torch.fft.irfft2(out_ft, s = (H, W))
#         return x

class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 傅里叶模态数（空间维度1）
        self.modes2 = modes2  # 傅里叶模态数（空间维度2）

        # 傅里叶权重参数（复数用两个实数表示）
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # 复数乘法操作 (batch, in_channel, x,y), (in_channel, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]

        # 傅里叶变换
        x_ft = torch.fft.rfft2(x)

        # 频域滤波
        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1,
                             device = x.device, dtype = torch.cfloat)

        # 处理低频模态
        modes1 = min(self.modes1, H)
        modes2 = min(self.modes2, W // 2 + 1)

        # 维度1的低频部分
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(x_ft[:, :, :modes1, :modes2],
            self.weights1[:, :, :modes1, :modes2])

        # 维度2的低频部分
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(x_ft[:, :, -modes1:, :modes2],
            self.weights2[:, :, -modes1:, :modes2])

        # 逆傅里叶变换
        x = torch.fft.irfft2(out_ft, s = (H, W))
        return x


def plot_lr_curve(learning_rates, save_path):
    """绘制学习率变化曲线并保存

    Args:
        learning_rates (list): 记录学习率的列表
        save_path (str): 图片保存路径
    """
    plt.figure(figsize = (10, 6))
    plt.plot(range(len(learning_rates)), learning_rates,
             label = 'Learning Rate', color = 'green', linewidth = 2)
    plt.xlabel('Training Steps', fontsize = 12)
    plt.ylabel('Learning Rate', fontsize = 12)
    plt.yscale('log')
    plt.title('Learning Rate Schedule Curve', fontsize = 14)
    plt.grid(True, linestyle = '--', alpha = 0.5)
    plt.legend()

    # 自动创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
    plt.close()

# 训练函数（保持结构不变，仅修改模型初始化）
def train(model, train_loader, val_loader, epochs, device):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    # 学习率调度器（保持不变）
    if args.lr_sch == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr = args.lr * 1.5,
            steps_per_epoch = len(train_loader),#len(train_loader)==batch数
            epochs = epochs, div_factor = 10.0, final_div_factor = 2.0)
    elif args.lr_sch == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.max_steps)  # 余弦退火

    else:
        scheduler = None

    best_loss = float('inf')

    progress_bar = tqdm(range(epochs), desc = "Training", ncols = 100)

    # 初始化列表来存储损失
    train_losses = []
    val_losses = []
    learning_rates = []

    # 创建日志目录
    os.makedirs('/data1/nl/FNOmain/fourier-neural-operator-main/log1/lr', exist_ok = True)

    for epoch in progress_bar:
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            if scheduler:
                # print(f"Epoch {epoch},  LR: {scheduler.get_last_lr()}")
                scheduler.step()
                learning_rates.append(optimizer.param_groups[0]['lr'])  # 新增行

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)

        # 计算平均损失
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        # 记录损失值
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"/data1/nl/FNOmain/fourier-neural-operator-main/best_model_fno6{timestamp}.pth")

        progress_bar.set_postfix({
            "Train Loss": f"{train_loss:.4e}",
            "Val Loss": f"{val_loss:.4e}"
        })

    # 修改保存路径包含时间戳
    loss_curve_path = f'/data1/nl/FNOmain/fourier-neural-operator-main/log1/loss/loss_curve_{timestamp}.png'
    summary_path = f'/data1/nl/FNOmain/fourier-neural-operator-main/log1/loss/training_summary_{timestamp}.txt'
    lr_curve_path = f'/data1/nl/FNOmain/fourier-neural-operator-main/log1/lr/lr_curve_{timestamp}.png'
    plot_lr_curve(learning_rates, lr_curve_path)

    # 绘制损失曲线
    plt.figure(figsize = (10, 6))
    plt.plot(range(epochs), train_losses, label = 'Train Loss', color = 'red')
    plt.plot(range(epochs), val_losses, label = 'Val Loss', color = 'blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Train and Validation Loss over Epochs')
    plt.legend()
    plt.savefig(loss_curve_path)
    plt.close()

    # 记录关键参数和最终Loss到文本文件
    final_train_loss = train_losses[-1] if len(train_losses) > 0 else 'N/A'
    final_val_loss = val_losses[-1] if len(val_losses) > 0 else 'N/A'

    with open(summary_path, 'w') as f:
        f.write(f"=== 训练摘要 ===\n")
        f.write(f"实验时间: {timestamp}\n\n")

        f.write(f"=== 关键参数 ===\n")
        f.write(f"傅里叶模态数 (modes): {args.modes1}\n")
        f.write(f"网络宽度 (width): {args.width}\n")
        f.write(f"最大训练步数 (max_steps): {args.max_steps}\n")
        f.write(f"学习率 (lr): {args.lr}\n")
        f.write(f"学习率调度器 (lr_sch): {args.lr_sch}\n\n")

        f.write(f"=== 最终结果 ===\n")
        f.write(f"最终训练Loss: {final_train_loss:.4e}\n")
        f.write(f"最终验证Loss: {final_val_loss:.4e}\n")

    print(f"训练完成！摘要已保存至: {summary_path}")
    # print("Training Complete!")
    return model


def visualize_predictions(model, val_loader, device, num_samples=None):
    """
    可视化测试集的真实解、预测解以及差值对比
    :param model: 训练好的模型
    :param val_loader: 测试集 DataLoader
    :param device: 设备 (cuda 或 cpu)
    """
    model.eval()
    sample_count = 0

    # 定义常量

    sigma = 40
    theta_init = 10
    theta_switch = -10

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if num_samples is not None and sample_count >= num_samples:  # 限制样本数量
                break

            # 逐个样本可视化
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # 转换为 numpy 数组
            input_np = inputs.cpu().numpy()[0]  # 取第一个样本 [C, H, W]
            target_np = targets.cpu().numpy()[0, 0]  # 真实解 [H, W]
            output_np = outputs.cpu().numpy()[0, 0]  # 预测解 [H, W]

            # 计算差值
            diff_np = np.abs(target_np - output_np)  # 真实解与预测解的差值

            # 提取 λ 值
            lambda_val = input_np[2, 0, 0]  # λ 是输入的第 3 个通道

            # 提取 x 和 t 的最小值和最大值
            x_min, x_max = np.min(input_np[0]), np.max(input_np[0])  # x 值的最小最大值
            t_min, t_max = np.min(input_np[1]), np.max(input_np[1])  # t 值的最小最大值

            # 可视化
            plt.figure(figsize=(15, 5))
            plt.suptitle(f"Test Sample {i + 1} | λ = {lambda_val:.2f}")

            # 真实解
            plt.subplot(1, 3, 1)
            plt.imshow(target_np, cmap='viridis', extent=[x_min, x_max, t_min, t_max])
            plt.title("True Solution")
            plt.colorbar()

            # 预测解
            plt.subplot(1, 3, 2)
            plt.imshow(output_np, cmap='viridis', extent=[x_min, x_max, t_min, t_max])
            plt.title("Predicted Solution")
            plt.colorbar()

            # 真实解与预测解差异
            plt.subplot(1, 3, 3)
            plt.imshow(diff_np, cmap='coolwarm', extent=[x_min, x_max, t_min, t_max])
            plt.title("Difference (True - Predicted)")
            plt.colorbar()
            plt.savefig(f'/data1/nl/FNOmain/fourier-neural-operator-main/log1/clog/{lambda_val}.png')
            plt.close()
            # plt.show()

            # 保存预测结果到 CSV 文件
            t_flat = np.linspace(t_min, t_max, output_np.shape[0])  # t 的值
            x_flat = np.linspace(x_min, x_max, output_np.shape[1])  # x 的值
            t_grid, x_grid = np.meshgrid(t_flat, x_flat, indexing = 'ij')  # 创建网格

            # 将 t, x, c 保存为 CSV 文件
            csv_filename = f"/data1/nl/FNOmain/fourier-neural-operator-main/data_generation/prediction_sample_{lambda_val}.csv"
            with open(csv_filename, mode = 'w', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow(['t', 'x', 'c'])  # 写入表头
                for t_idx in range(len(t_flat)):
                    for x_idx in range(len(x_flat)):
                        writer.writerow([
                            t_grid[t_idx, x_idx],  # t 值
                            x_grid[t_idx, x_idx],  # x 值
                            output_np[t_idx, x_idx]  # c 值
                        ])
            print(f"预测结果已保存到 {csv_filename}")

            # 保存预测结果到 CSV 文件
            t_flat = np.linspace(t_min, t_max, target_np.shape[0])  # t 的值
            x_flat = np.linspace(x_min, x_max, target_np.shape[1])  # x 的值
            t_grid, x_grid = np.meshgrid(t_flat, x_flat, indexing = 'ij')  # 创建网格

            # 将 t, x, c 保存为 CSV 文件
            csv_filename = f"/data1/nl/FNOmain/fourier-neural-operator-main/data_generation/prediction_sample_target_{lambda_val}.csv"
            with open(csv_filename, mode = 'w', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow(['t', 'x', 'c'])  # 写入表头
                for t_idx in range(len(t_flat)):
                    for x_idx in range(len(x_flat)):
                        writer.writerow([
                            t_grid[t_idx, x_idx],  # t 值
                            x_grid[t_idx, x_idx],  # x 值
                            target_np[t_idx, x_idx]  # c 值
                        ])
            print(f"预测结果已保存到 {csv_filename}")


            t_sim=t_max
            # 计算 flux
            x_sim = lambda_val * np.sqrt(t_sim)
            t_flat = np.linspace(0, t_sim, output_np.shape[0])
            x_flat = np.linspace(0, x_sim, output_np.shape[1])
            dx = x_flat[1] - x_flat[0]  # 计算 Δx

            # 计算 flux
            flux = -(output_np[:, 1] - output_np[:, 0]) / dx

            # 计算 potential
            cv_flat = np.where(t_flat < t_sim / 2.0, theta_init - sigma * t_flat, theta_switch + sigma * (t_flat - t_sim / 2.0))

            # 绘制 flux 曲线
            plt.figure(figsize=(10, 6))
            plt.plot(cv_flat, flux, linestyle='--', color='red',label='predict')

            t_flat1 = np.linspace(0, t_sim, target_np.shape[0])
            x_flat1 = np.linspace(0, x_sim, target_np.shape[1])
            dx1 = x_flat1[1] - x_flat1[0]  # 计算 Δx
            flux1 = -(target_np[:, 1] - target_np[:, 0]) / dx1
            cv_flat1 = np.where(t_flat1 < t_sim / 2.0, theta_init - sigma * t_flat1,
                                theta_switch + sigma * (t_flat1 - t_sim / 2.0))
            plt.plot(cv_flat1, flux1, linestyle = '--', color = 'blue',label='true')

            # plt.plot(cv_flat, flux, label='Finite Difference Method', linestyle='--', color='red')
            plt.xlabel("Potential")
            plt.ylabel("Flux")
            plt.legend()
            plt.title(f"Flux vs Potential for Sample {i + 1} | λ = {lambda_val:.2f}")
            plt.savefig(f'/data1/nl/FNOmain/fourier-neural-operator-main/log1/cvlog/{lambda_val}.png')
            # plt.show()
            # # 绘制 flux 曲线
            # plt.figure(figsize = (10, 6))

            # plt.show()
            plt.close()
            sample_count += 1
# 新增函数：根据λ值动态生成输入网格
def generate_input_grid(lambda_val, t_resolution = 101, x_resolution = 200):
    """
    根据λ值动态生成输入网格
    :param lambda_val: 用户指定的λ值
    :param t_resolution: t方向的分辨率
    :param x_resolution: x方向的分辨率
    :return: 输入网格 [3, H, W]，其中通道顺序为 [x, t, λ]
    """
    # 生成t和x的网格
    t = np.linspace(0, 1, t_resolution)  # t范围固定为[0, 1]
    x = np.linspace(0, lambda_val, x_resolution)  # x范围动态调整为[0, λ]

    # 创建网格
    t_grid, x_grid = np.meshgrid(t, x, indexing = 'ij')

    # 创建λ网格（全为λ值）
    lambda_grid = np.full_like(x_grid, lambda_val)

    # 组合成输入网格 [3, H, W]
    input_grid = np.stack([x_grid, t_grid, lambda_grid], axis = 0)

    # 转换为PyTorch张量
    return torch.tensor(input_grid, dtype = torch.float32)


# 修改后的预测函数
def predict_lambda(model, lambda_val, device, t_resolution = 101, x_resolution = 200):
    """
    根据λ值生成预测结果
    :param model: 训练好的模型
    :param lambda_val: 用户指定的λ值
    :param device: 计算设备
    :param t_resolution: t方向的分辨率
    :param x_resolution: x方向的分辨率
    :return: 预测结果 [H, W]
    """
    # 动态生成输入网格
    input_grid = generate_input_grid(lambda_val, t_resolution, x_resolution)

    # 添加batch维度并传输到设备
    input_tensor = input_grid.unsqueeze(0).to(device)

    # 模型预测
    with torch.no_grad():
        prediction = model(input_tensor)

    # 返回预测结果（去除batch维度）
    return prediction.squeeze().cpu().numpy()


# 修改后的交互式预测界面
def interactive_predict(model, device, t_resolution = 101, x_resolution = 200):
    """
    交互式预测界面
    :param model: 训练好的模型
    :param device: 计算设备
    :param t_resolution: t方向的分辨率
    :param x_resolution: x方向的分辨率
    """
    # 定义物理参数（根据实际问题设置）
    sigma = 40  # 温度变化率
    theta_init = 10  # 初始温度
    theta_switch = -10  # 切换后的温度

    while True:
        try:
            # 获取用户输入
            user_input = input("请输入λ值（输入q退出）: ")
            if user_input.lower() == 'q':
                break

            lambda_val = float(user_input)

            # 生成预测
            prediction = predict_lambda(model, lambda_val, device, t_resolution, x_resolution)

            # --- 可视化部分 ---
            plt.figure(figsize = (20, 6))

            # 子图1: xtc热图
            plt.subplot(1, 2, 1)
            plt.imshow(prediction, cmap = 'viridis', extent = [0, lambda_val, 0, 1])
            plt.title(f"Predicted c(t,x) for λ={lambda_val:.2f}")
            plt.xlabel('x')
            plt.ylabel('t')
            plt.colorbar(label = 'c value')

            # 子图2: 预测解的flux曲线
            plt.subplot(1, 2, 2)

            # 计算flux相关参数
            t_sim = 1.0  # 假设总时间固定为1.0
            x_sim = lambda_val
            t_flat = np.linspace(0, t_sim, prediction.shape[0])
            x_flat = np.linspace(0, x_sim, prediction.shape[1])

            # 计算flux（基于预测解）
            dx = x_flat[199] - x_flat[20] if len(x_flat) > 20 else 0.1  # 防止索引越界
            flux = -(prediction[:, 199] - prediction[:, 20]) / dx if prediction.shape[1] > 20 else np.zeros_like(t_flat)
            cv_flat = np.where(t_flat < t_sim / 2.0,
                               theta_init - sigma * t_flat,
                               theta_switch + sigma * (t_flat - t_sim / 2.0))

            # 绘制曲线
            plt.plot(cv_flat, flux, 'r--', label = 'Predicted')
            plt.xlabel("Potential")
            plt.ylabel("Flux")
            plt.title("Predicted Flux vs Potential")
            plt.legend()

            # 保存图像
            save_dir = "/data1/nl/FNOmain/fourier-neural-operator-main/log1/interactive/"
            os.makedirs(save_dir, exist_ok = True)
            plt.savefig(f"{save_dir}prediction_lambda_{lambda_val:.2f}.png")
            plt.show()

        except ValueError:
            print("输入错误，请输入有效的数字！")
        except Exception as e:
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    # 数据路径配置（根据实际情况修改）
    train_dir = '/data1/nl/FNOmain/fourier-neural-operator-main/data6/train'
    val_dir = '/data1/nl/FNOmain/fourier-neural-operator-main/data6/val'
    test_dir = '/data1/nl/FNOmain/fourier-neural-operator-main/data6/test'

    train_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.pt')]
    val_paths = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.pt')]
    test_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.pt')]

    # 创建数据加载器
    train_dataset = PDEDataset(train_paths)
    val_dataset = PDEDataset(val_paths)
    test_dataset = PDEDataset(test_paths)

    train_loader = DataLoader(train_dataset, batch_size = len(train_dataset), shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = len(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size = 1)

    # 初始化FNO模型
    model = FNO2d(
        in_channels = 3,
        out_channels = 1,
        modes1 = args.modes1,
        modes2 = args.modes2,
        width = args.width
    )
    model.to(device)
    # # 训练模型
    trained_model = train(model, train_loader, val_loader, epochs = args.max_steps, device = device)
    print("model saved")

    # 加载最佳模型并可视化
    model.load_state_dict(torch.load(f"/data1/nl/FNOmain/fourier-neural-operator-main/best_model_fno6{timestamp}.pth", map_location = device))
    visualize_predictions(model, test_loader, device)

    # model.load_state_dict(torch.load("/data1/nl/FNOmain/fourier-neural-operator-main/best_model_fno20250320_153206.pth", map_location = device))
    # visualize_predictions(model, test_loader, device)


    # # 加载最佳模型
    # model.load_state_dict(torch.load("/data1/nl/FNOmain/fourier-neural-operator-main/best_model_fno1.pth", map_location = device))
    # model.eval()
    #
    # 启动交互式预测界面
    print("\n进入预测模式：")
    interactive_predict(model, device)