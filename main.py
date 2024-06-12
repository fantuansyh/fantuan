import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as pyg_nn
#from torch_geometric.data import DataLoader
# 更新 DataLoader 的导入路径
from torch_geometric.loader import DataLoader

from datafirstdeal import load_or_create_graph_data_list


root_dir = 'data/24.5.13 pdb_dataset'
subdirs = ['label_0', 'label_1', 'label_2']
# 假设 graph_data_list 包含所有你提取的图数据对象

save_path = 'E:/iGEM/前期准备/pythonProject/data/processed_graph_data.pkl'  # 保存路径

graph_data_list = load_or_create_graph_data_list(root_dir, subdirs, save_path)

#graph_data_list = create_graph_data_list(root_dir, subdirs)

# 打乱数据并划分训练集和测试集
test_ratio = 0.15  # 测试集比例
num_samples = len(graph_data_list)
indices = torch.randperm(num_samples)
graph_data_list = [graph_data_list[i] for i in indices]
print(f"Loaded {len(graph_data_list)} graphs.")

split_index = int(num_samples * (1 - test_ratio))
train_dataset = graph_data_list[:split_index]
test_dataset = graph_data_list[split_index:]

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)# 定义 GAT 模型
class Net(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(Net, self).__init__()
        self.conv1 = pyg_nn.GATConv(num_node_features, 16)  # 第一层，特征维数从输入维度到16
        self.conv2 = pyg_nn.GATConv(16, 32)                 # 第二层，特征维数从16到32
        self.conv3 = pyg_nn.GATConv(32, 64)                 # 新增的第三层，特征维数从32到64
        self.fc = nn.Linear(64, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)  # 新增层的应用
        x = F.relu(x)
        x = pyg_nn.global_max_pool(x, batch)

        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# 获取节点和类别信息
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_node_features = train_dataset[0].x.size(1)
num_classes = len(set(data.y.item() for data in train_dataset + test_dataset))


# num_label0 = 4651
# num_label1 = 891
# num_label2 = 1844
# total_samples = num_label0 + num_label1 + num_label2
# weight_label0 = total_samples / (num_classes * num_label0)
# weight_label1 = total_samples / (num_classes * num_label1)
# weight_label2 = total_samples / (num_classes * num_label2)
# class_weights = torch.tensor([weight_label0, weight_label1, weight_label2], dtype=torch.float32).to(device)# 初始化模型、优化器和损失函数

model = Net(num_node_features, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.NLLLoss()#weight=class_weights

# 训练模型
model.train()
epochs = 200
for epoch in range(epochs):
    correct_train = 0
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)

        loss = loss_function(pred, data.y)
        total_loss += loss.item()
        correct_train += pred.argmax(axis=1).eq(data.y).sum().item()

        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        train_acc = correct_train / len(train_dataset)
        train_loss = total_loss / len(train_dataset)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')

# 测试模型并打印混淆矩阵
model.eval()
num_classes = len(set(data.y.item() for data in test_dataset))
confusion_matrix = torch.zeros(num_classes, num_classes)

with torch.no_grad():
    correct_test = 0
    total_loss_test = 0

    for data in test_loader:
        data = data.to(device)
        pred = model(data)

        loss = loss_function(pred, data.y)
        total_loss_test += loss.item()
        correct_test += pred.argmax(axis=1).eq(data.y).sum().item()

        pred_labels = pred.argmax(axis=1)
        for true, predicted in zip(data.y.view(-1), pred_labels.view(-1)):
            confusion_matrix[true.long(), predicted.long()] += 1

    test_acc = correct_test / len(test_dataset)
    test_loss = total_loss_test / len(test_dataset)
    print(f'Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
    print("Confusion Matrix:")
    print(confusion_matrix)
