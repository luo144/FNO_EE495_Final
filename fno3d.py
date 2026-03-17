import csv

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 参数解析
parser = argparse.ArgumentParser()

parser.add_argument('--max_steps', type = int, default = 300)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--lr_sch', type = str, default = 'onecycle')
parser.add_argument('--modes1', type = int, default = 24)  # 傅里叶模态数
parser.add_argument('--modes2', type = int, default = 24)  # 傅里叶模态数
parser.add_argument('--modes3', type = int, default = 24)  # 傅里叶模态数
parser.add_argument('--width', type = int, default = 32)  # 网络宽度
args = parser.parse_args()



class PDEDataset3D(torch.utils.data.Dataset):
    def __init__(self, merged_dir):
        """加载包含文件名的合并数据"""
        self.merged_files = sorted(Path(merged_dir).glob("merged_*.pt"))
        self.samples = []

        # 加载所有合并文件
        for f in self.merged_files:
            chunk = torch.load(f, map_location = 'cpu')
            # 每个样本格式为 (data, filename)
            self.samples.extend(chunk)  # 直接扩展样本列表

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 返回格式：(input_tensor, target_tensor, filename)
        data, filename = self.samples[idx]
        return data[0], data[1], filename  # 假设原始data是(input, target)


def validate_merge(original_dir, merged_dir):
    """验证合并后的数据完整性（支持tuple比较）"""
    orig_files = sorted(Path(original_dir).glob("*.pt"))
    merged_samples = len(PDEDataset3D(merged_dir))

    print(f"\n验证结果：")
    print(f"原始文件数：{len(orig_files)}")
    print(f"合并后样本总数：{merged_samples}")
    assert len(orig_files) == merged_samples, "样本数量不一致！"

    # 随机抽查样本
    import random
    test_idx = random.randint(0, merged_samples - 1)

    # 加载原始数据
    orig_data = torch.load(orig_files[test_idx])

    # 加载合并数据
    merged_data = PDEDataset3D(merged_dir)[test_idx]

    # 类型校验
    assert type(orig_data) == type(merged_data), f"类型不匹配: {type(orig_data)} vs {type(merged_data)}"

    # 分情况比较
    if isinstance(orig_data, torch.Tensor):
        assert torch.allclose(orig_data, merged_data), f"张量数据不匹配"
    elif isinstance(orig_data, tuple):
        assert len(orig_data) == len(merged_data), f"元组长度不同"
        for i, (orig, merged) in enumerate(zip(orig_data, merged_data)):
            assert torch.allclose(orig, merged), f"元组第{i}个元素不匹配"
    else:
        raise TypeError(f"不支持的数据类型: {type(orig_data)}")

    print("随机样本校验通过！")


# FNO模型定义
class FNO3d(nn.Module):
    def __init__(self, in_channels = 9, out_channels = 1, modes1 = 8, modes2 = 8, modes3 = 8, width = 32):
        super(FNO3d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width

        # 输入投影 (9通道 -> width)
        self.p = nn.Conv3d(in_channels, self.width, 1)

        # 傅里叶层堆叠
        self.fourier_layers = nn.ModuleList([
            FourierLayer3D(self.width, self.width, modes1, modes2, modes3)
            for _ in range(4)
        ])

        # 局部卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv3d(self.width, self.width, 1) for _ in range(4)
        ])

        # 输出投影 (width -> 1通道)
        self.q = nn.Conv3d(self.width, out_channels, 1)

    def forward(self, x):
        # 输入形状: [batch, 9, X, Y, T]
        x = self.p(x)

        for f_layer, c_layer in zip(self.fourier_layers, self.conv_layers):
            x_f = f_layer(x)
            x_c = c_layer(x)
            x = x_f + x_c  # 残差连接
            x = F.gelu(x)

        x = self.q(x)  # [batch, 1, X, Y, T]
        return x


class FourierLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        # 复数权重初始化
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype = torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype = torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype = torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype = torch.cfloat)
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # 正向实数FFT（获得复数频谱）
        x_ft = torch.fft.rfftn(x, dim = [-3, -2, -1], norm = "ortho")

        # 初始化复数输出张量
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype = torch.cfloat, device = x.device)

        # 区域1计算
        modes1 = min(self.modes1, x_ft.size(-3))
        modes2 = min(self.modes2, x_ft.size(-2))
        modes3 = min(self.modes3, x_ft.size(-1))

        out_ft[..., :modes1, :modes2, :modes3] = self.compl_mul3d(
            x_ft[..., :modes1, :modes2, :modes3],
            self.weights1[..., :modes1, :modes2, :modes3]
        )
        out_ft[..., -modes1:, :modes2, :modes3] = self.compl_mul3d(
            x_ft[..., -modes1:, :modes2, :modes3],
            self.weights2[..., -modes1:, :modes2, :modes3]
        )
        out_ft[..., :modes1, -modes2:, :modes3] = self.compl_mul3d(
            x_ft[..., :modes1, -modes2:, :modes3],
            self.weights3[..., :modes1, -modes2:, :modes3]
        )
        out_ft[..., -modes1:, -modes2:, :modes3] = self.compl_mul3d(
            x_ft[..., -modes1:, -modes2:, :modes3],
            self.weights4[..., -modes1:, -modes2:, :modes3]
        )

        # 逆FFT变换
        x = torch.fft.irfftn(out_ft, s = (x.size(-3), x.size(-2), x.size(-1)), norm = "ortho")
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
            steps_per_epoch = len(train_loader),
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
    os.makedirs('/data1/nl/FNOmain/fourier-neural-operator-main/log3/lr', exist_ok = True)


    for epoch in progress_bar:
        model.train()
        train_loss = 0.0

        for inputs, targets,_ in train_loader:
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
        # if args.lr_sch == 'cos':
        #     scheduler.step()

        # 验证过程
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets,_ in val_loader:
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
            torch.save(model.state_dict(), f"/data1/nl/FNOmain/fourier-neural-operator-main/best_model_fno3{timestamp}.pth")

        progress_bar.set_postfix({
            "Train Loss": f"{train_loss:.4e}",
            "Val Loss": f"{val_loss:.4e}"
        })

    # 修改保存路径包含时间戳
    loss_curve_path = f'/data1/nl/FNOmain/fourier-neural-operator-main/log3/loss/loss_curve_{timestamp}.png'
    summary_path = f'/data1/nl/FNOmain/fourier-neural-operator-main/log3/loss/training_summary_{timestamp}.txt'
    lr_curve_path = f'/data1/nl/FNOmain/fourier-neural-operator-main/log3/lr/lr_curve_{timestamp}.png'
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
        f.write(f"傅里叶模态数 (modes1): {args.modes1}\n")
        f.write(f"傅里叶模态数 (modes2): {args.modes2}\n")
        f.write(f"傅里叶模态数 (modes3): {args.modes3}\n")
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

def visualize_3d_predictions(model, test_loader, device, save_dir, num_samples = None, plot_type = 'surface',
                             sample_ratio = 0.1):
    """
    整合后的三维可视化与结果保存
    :param model: 训练好的模型
    :param test_loader: 测试集DataLoader（需返回文件名）
    :param save_dir: 结果保存目录（自动创建plots和csv子目录）
    :param plot_type: 'surface'或'scatter'（默认曲面图）
    :param sample_ratio: 数据采样率（0~1）
    """
    model.eval()
    os.makedirs(f"{save_dir}/plots", exist_ok = True)
    os.makedirs(f"{save_dir}/csv", exist_ok = True)

    with torch.no_grad():
        for batch_idx, (inputs, targets, filenames) in enumerate(test_loader):
            if num_samples is not None and batch_idx >= num_samples:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            for i in range(inputs.size(0)):
                orig_filename = os.path.splitext(filenames[i])[0]
                csv_path = f"{save_dir}/csv/{orig_filename}_pred.csv"

                # === 保存CSV ===
                save_3d_data(
                    x = inputs[i, 1].cpu().numpy(),
                    y = inputs[i, 2].cpu().numpy(),
                    t = inputs[i, 0].cpu().numpy(),
                    true = targets[i, 0].cpu().numpy(),
                    pred = outputs[i, 0].cpu().numpy(),
                    csv_path = csv_path
                )

                # === 静态三维图 ===
                plot_3d_from_csv(
                    csv_path = csv_path,
                    save_dir = f"{save_dir}/plots",
                    plot_type = plot_type,
                    sample_ratio = sample_ratio
                )

def save_3d_data(x, y, t, true, pred, csv_path):
    """保存三维数据到CSV（兼容原逻辑）"""
    with open(csv_path, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['X', 'Y', 'Time', 'True', 'Predicted', 'Error'])
        for xi in range(x.shape[0]):
            for yi in range(x.shape[1]):
                for ti in range(t.shape[2]):
                    row = [
                        x[xi, yi, ti],
                        y[xi, yi, ti],
                        t[xi, yi, ti],
                        true[xi, yi, ti],
                        pred[xi, yi, ti],
                        abs(true[xi, yi, ti] - pred[xi, yi, ti])
                    ]
                    writer.writerow(row)
    print(f"CSV saved: {csv_path}")


def plot_3d_from_csv(csv_path, save_dir, plot_type = 'surface', sample_ratio = 0.1):
    """基于CSV的三维可视化（新增函数）"""
    # 加载并预处理数据
    df = pd.read_csv(csv_path)
    if sample_ratio < 1.0:
        df = df.sample(frac = sample_ratio, random_state = 42)

    # 获取全局颜色范围
    cmin = min(df['True'].min(), df['Predicted'].min())
    cmax = max(df['True'].max(), df['Predicted'].max())
    cmax_error = df['Error'].max()

    # 创建画布并调整布局
    fig = plt.figure(figsize=(20, 7))  # 增加画布宽度
    fig.suptitle(os.path.basename(csv_path).replace('_pred.csv', ''), fontsize=14)
    plt.subplots_adjust(left=0.05, right=0.92, wspace=0.3, bottom=0.1)  # 增加间距

    # 定义通用绘图函数
    # 定义统一绘图函数
    def plot_subplot(ax, z_col, cmap, title, is_error = False):
        # 设置坐标轴字体
        ax.tick_params(labelsize = 10)
        ax.set_xlabel('X', fontsize = 12, labelpad = 10)
        ax.set_ylabel('Y', fontsize = 12, labelpad = 10)
        ax.set_zlabel('T', fontsize = 12, labelpad = 10)

        if plot_type == 'surface':
            x_unique = np.sort(df['X'].unique())
            y_unique = np.sort(df['Y'].unique())
            t_unique = np.sort(df['Time'].unique())
            X, Y, T = np.meshgrid(x_unique, y_unique, t_unique, indexing = 'ij')
            grid = df.pivot_table(index = ['X', 'Y', 'Time'], values = z_col).values.reshape(X.shape)
            t_mid = len(t_unique) // 2
            surf = ax.plot_surface(X[:, :, t_mid], Y[:, :, t_mid], grid[:, :, t_mid], cmap = cmap)
            fig.colorbar(surf, ax = ax, label = z_col)
        else:
            if is_error:
                sc = ax.scatter(df['X'], df['Y'], df['Time'],
                                c = df[z_col], cmap = 'coolwarm',
                                vmin = 0, vmax = cmax_error,  # 误差单独设置
                                s = 3, alpha = 0.8)
            else:
                sc = ax.scatter(df['X'], df['Y'], df['Time'],
                                c = df[z_col], cmap = 'viridis',
                                vmin = cmin, vmax = cmax,  # 统一色标范围
                                s = 3, alpha = 0.8)
            # 调整颜色条位置和大小
            cbar = fig.colorbar(sc, ax = ax, shrink = 0.6, pad = 0.1)
            cbar.ax.tick_params(labelsize = 10)

        ax.set_title(title, fontsize = 13, pad = 15)

    # 绘制三个子图
    plot_subplot(fig.add_subplot(131, projection = '3d'), 'True', 'viridis', "True")
    plot_subplot(fig.add_subplot(132, projection = '3d'), 'Predicted', 'viridis', "Predicted")
    plot_subplot(fig.add_subplot(133, projection = '3d'), 'Error', 'coolwarm', "Error",is_error=True)

    # 保存图像
    plot_filename = os.path.basename(csv_path).replace('_pred.csv', f'_{plot_type}.png')
    plt.savefig(f"{save_dir}/{plot_filename}", dpi = 150, bbox_inches = 'tight')
    plt.close()
    print(f"三维图保存至: {save_dir}/{plot_filename}")


def benchmark_loader(loader, name):
    import time
    start = time.perf_counter()
    for _ in loader:
        pass
    duration = time.perf_counter() - start
    print(f"{name} 加载耗时：{duration:.2f}秒 | 平均每batch {duration/len(loader):.3f}秒")


if __name__ == "__main__":
    num_workers=24
    prefetch_factor=4
    persistent_workers = True  # 保持worker进程存活

    # # 数据路径配置（根据实际情况修改）
    train_dir = '/data1/nl/FNOmain/fourier-neural-operator-main/data4/train_merged'
    val_dir = '/data1/nl/FNOmain/fourier-neural-operator-main/data4/val_merged'
    test_dir = '/data1/nl/FNOmain/fourier-neural-operator-main/data4/test_merged'
    # 数据路径配置（根据实际情况修改）
    # train_dir = '/data1/nl/FNOmain/fourier-neural-operator-main/data4/train_merged_1'
    # val_dir = '/data1/nl/FNOmain/fourier-neural-operator-main/data4/train_merged_1'
    # test_dir = '/data1/nl/FNOmain/fourier-neural-operator-main/data4/train_merged_1'


    # train_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.pt')]
    # val_paths = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.pt')]
    # test_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.pt')]

    # # 创建数据加载器
    # train_dataset = PDEDataset3D(train_paths)
    # val_dataset = PDEDataset3D(val_paths)
    # test_dataset = PDEDataset3D(test_paths)

    # # 执行验证
    # validate_merge(
    #     original_dir = "/data1/nl/FNOmain/fourier-neural-operator-main/data4/train",
    #     merged_dir = "/data1/nl/FNOmain/fourier-neural-operator-main/data4/train_merged"
    # )
    # 创建数据集（无需再收集文件路径）
    train_dataset = PDEDataset3D(train_dir)
    val_dataset = PDEDataset3D(val_dir)
    test_dataset = PDEDataset3D(test_dir)


    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True,num_workers = num_workers,prefetch_factor = prefetch_factor,pin_memory=True,persistent_workers = persistent_workers)
    val_loader = DataLoader(val_dataset, batch_size = 32,num_workers=num_workers // 2,  pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size = 32,num_workers=num_workers // 2,  pin_memory = True)

    # # 在训练前调用
    # benchmark_loader(train_loader, "训练集")
    # benchmark_loader(val_loader, "验证集")
    # 初始化FNO模型
    model = FNO3d(
        in_channels = 9,
        out_channels = 1,
        modes1 = args.modes1,
        modes2 = args.modes2,
        modes3 = args.modes3,
        width = args.width
    )
    model.to(device)
    # # 训练模型
    # trained_model = train(model, train_loader, val_loader, epochs = args.max_steps, device = device)
    # print("model saved")

    # # 加载最佳模型并可视化
    model.load_state_dict(torch.load("/data1/nl/FNOmain/fourier-neural-operator-main/best_model_fno320250320_153934.pth", map_location = device))
    # model.eval()
    # # 调用示例
    visualize_3d_predictions(model=model, test_loader=test_loader, device=device,  save_dir = "/data1/nl/FNOmain/fourier-neural-operator-main/log3/predictions1",
                             plot_type='scatter', sample_ratio=1.0,num_samples = 1)


    # # 启动交互式预测界面
    # print("\n进入预测模式：")
    # interactive_predict(model, device)
