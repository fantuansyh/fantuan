import torch
import torch.nn as nn

# 检查CUDA是否可用
print("CUDA Available:", torch.cuda.is_available())

# 简单的网络定义
class Net(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_node_features, 100)  # 示例层
        self.fc2 = nn.Linear(100, num_classes)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 假定的特征和类别数量
num_node_features = 10
num_classes = 3

# 创建并移动模型到CUDA
if torch.cuda.is_available():
    model = Net(num_node_features, num_classes).to("cuda")
    print("Model moved to CUDA")
else:
    print("CUDA not available, model stays on CPU")
