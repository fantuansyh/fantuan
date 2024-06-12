import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
from datafirstdeal import load_or_create_graph_data_list
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    root_dir = 'data/24.5.13 pdb_dataset'
    subdirs = ['label_0', 'label_1', 'label_2']
    save_path = 'E:/iGEM/前期准备/pythonProject/data/processed_graph_data.pkl'

    graph_data_list = load_or_create_graph_data_list(root_dir, subdirs, save_path)

    # 获取每个类别的样本数量
    label_counter = Counter([data.y.item() for data in graph_data_list])
    max_label_count = max(label_counter.values())

    # 定义 device 变量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 计算每个类别的权重
    num_classes = len(label_counter)
    initial_weights = [max_label_count / label_counter[i] for i in range(num_classes)]

    # 调整权重，根据实验结果进行调整
    class_weights = torch.tensor([initial_weights[0], initial_weights[1] * 1.2, initial_weights[2] * 1.2],
                                 dtype=torch.float).to(device)

    # 创建 WeightedRandomSampler
    weights = [class_weights[data.y.item()] for data in graph_data_list]
    sampler = WeightedRandomSampler(weights, num_samples=len(graph_data_list), replacement=True)

    # 打乱数据并划分训练集和测试集
    test_ratio = 0.15
    num_samples = len(graph_data_list)
    indices = torch.randperm(num_samples)
    graph_data_list = [graph_data_list[i] for i in indices]
    print(f"Loaded {len(graph_data_list)} graphs after balancing.")

    split_index = int(num_samples * (1 - test_ratio))
    train_dataset = graph_data_list[:split_index]
    test_dataset = graph_data_list[split_index:]

    # 打印训练集和测试集的长度
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # 创建数据加载器，使用 WeightedRandomSampler 进行过采样
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=WeightedRandomSampler(
        [max_label_count / label_counter[data.y.item()] for data in train_dataset], num_samples=len(train_dataset),
        replacement=True), num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

    # 定义 GAT 模型
    class Net(nn.Module):
        def __init__(self, num_node_features, num_classes):
            super(Net, self).__init__()
            self.conv1 = pyg_nn.GATConv(num_node_features, 16)
            self.conv2 = pyg_nn.GATConv(16, 32)
            self.conv3 = pyg_nn.GATConv(32, 64)
            self.fc = nn.Linear(64, num_classes)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = pyg_nn.global_max_pool(x, batch)
            x = self.fc(x)
            return F.log_softmax(x, dim=1)

    # 初始化模型、优化器和损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_node_features = train_dataset[0].x.size(1)
    num_classes = len(set(data.y.item() for data in train_dataset + test_dataset))

    model = Net(num_node_features, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.NLLLoss(weight=class_weights)

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

    # 测试模型并打印混淆矩阵和评估指标
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        correct_test = 0
        total_loss_test = 0

        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            loss = loss_function(pred, data.y)
            total_loss_test += loss.item()
            correct_test += pred.argmax(axis=1).eq(data.y).sum().item()
            all_preds.extend(pred.argmax(axis=1).cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            pred_labels = pred.argmax(axis=1)
            for true, predicted in zip(data.y.view(-1), pred_labels.view(-1)):
                confusion_matrix[true.long(), predicted.long()] += 1

        test_acc = correct_test / len(test_dataset)
        test_loss = total_loss_test / len(test_dataset)
        print(f'Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
        print("Confusion Matrix:")
        print(confusion_matrix)

    # 计算 F1-score、Precision 和 Recall
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    print(f'F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')


if __name__ == '__main__':
    main()
