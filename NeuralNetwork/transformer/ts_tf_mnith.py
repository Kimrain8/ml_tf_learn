import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math
import time

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- 数据准备 --------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# -------------------- Vision Transformer 模型 --------------------
class PatchEmbedding(nn.Module):
    """将图像分割成 patches 并映射为嵌入向量"""
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 4x4=16

        # 使用卷积实现 patch 分割和线性投影 (kernel_size = patch_size, stride = patch_size)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)                # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2)                 # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)            # [B, num_patches, embed_dim]
        return x

class TransformerEncoderBlock(nn.Module):
    """单个 Transformer 编码器块"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 自注意力 + 残差
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # 前馈 + 残差
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        return x

class VisionTransformer(nn.Module):
    """用于 MNIST 分类的 Vision Transformer"""
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10,
                 embed_dim=64, depth=4, num_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()

        # Patch 嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 可学习的类别令牌 (class token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer 编码器
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 按照 ViT 的常见初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        # 获取 patch 嵌入
        x = self.patch_embed(x)          # [B, num_patches, embed_dim]

        # 添加类别令牌
        cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)           # [B, num_patches+1, embed_dim]

        # 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)

        # 通过 Transformer 块
        for block in self.blocks:
            x = block(x)

        # 提取类别令牌对应的输出
        x = self.norm(x)
        cls_out = x[:, 0]                 # [B, embed_dim]
        logits = self.head(cls_out)       # [B, num_classes]
        return logits

# -------------------- 训练和评估 --------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

# -------------------- 主训练循环 --------------------
model = VisionTransformer(
    img_size=28,
    patch_size=7,
    in_channels=1,
    num_classes=10,
    embed_dim=64,
    depth=4,
    num_heads=4,
    ff_dim=128,
    dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

num_epochs = 20
best_acc = 0

print("开始训练...")
for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    scheduler.step()

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch:2d}/{num_epochs} | Time: {epoch_time:.1f}s | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "best_mnist_vit.pth")
        print(f"  -> 保存最佳模型 (准确率: {best_acc:.2f}%)")

print(f"\n训练完成！最佳测试准确率: {best_acc:.2f}%")