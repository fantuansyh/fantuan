%% 清空环境变量
warning off;             % 关闭报警信息
close all;               % 关闭开启的图窗
clear;                   % 清空变量
clc;                     % 清空命令行
tic;

%% 划分训练集和测试集
% 假设已经定义了one_hot1, one_hot2, one_hot3, one_hot4, one_hot5
trainP = one_hot1; % 类别1的训练数据
trainPosmerge = one_hot2; % 类别2的训练数据
trainN = one_hot3; % 类别3的训练数据
testPos = one_hot4; % 类别1的测试数据
testNeg = one_hot5; % 类别2的测试数据
% 假设 one_hot6 是类别3的测试数据
testAnother = one_hot6; % 示例，你需要根据实际情况调整


% 计算最大列数
maxCols = max([size(trainP, 2), size(trainPosmerge, 2), size(trainN, 2), size(testPos, 2), size(testNeg, 2), size(testAnother, 2)]);

% 填充零以匹配最大列数
trainP_padded = [trainP, zeros(size(trainP, 1), maxCols - size(trainP, 2))];
trainPosmerge_padded = [trainPosmerge, zeros(size(trainPosmerge, 1), maxCols - size(trainPosmerge, 2))];
trainN_padded = [trainN, zeros(size(trainN, 1), maxCols - size(trainN, 2))];
testPos_padded = [testPos, zeros(size(testPos, 1), maxCols - size(testPos, 2))];
testNeg_padded = [testNeg, zeros(size(testNeg, 1), maxCols - size(testNeg, 2))];
testAnother_padded = [testAnother, zeros(size(testAnother, 1), maxCols - size(testAnother, 2))];

% 进行垂直连接
P_train = [trainP_padded; trainPosmerge_padded; trainN_padded]';
T_train = [ones(size(trainP_padded, 1), 1); 2 * ones(size(trainPosmerge_padded, 1), 1); 3 * ones(size(trainN_padded, 1), 1)]';
M = size(P_train, 2);

P_test = [testPos_padded; testNeg_padded; testAnother_padded]';
T_test = [ones(size(testPos_padded, 1), 1); 2 * ones(size(testNeg_padded, 1), 1); 3 * ones(size(testAnother_padded, 1), 1)]';
N = size(P_test, 2);
%% 数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test = mapminmax('apply', P_test, ps_input);

t_train = categorical(T_train)';
t_test = categorical(T_test)';

% 假设P_train是一个N*M的矩阵，其中N是特征数量，M是样本数量
% 我们将它转换为一个1*M的元胞数组，每个元胞包含一个N*1的矩阵

P_train_cell = num2cell(P_train, 1); % 将矩阵转换为元胞数组，每个元胞包含一个序列
t_train_cell = num2cell(t_train, 1); % 同理，对标签进行转换

% 注意：对于分类问题，标签通常不需要是序列，但如果你的网络设计要求标签也是序列，则应执行此步骤



%% 创建网络
layers = [ ...
  sequenceInputLayer(3760)              % 输入层
  lstmLayer(6, 'OutputMode', 'last')   % LSTM层
  reluLayer                            % Relu激活层
  fullyConnectedLayer(3)               % 全连接层，适应三分类
  softmaxLayer                         % 分类层
  classificationLayer];

%% 参数设置
options = trainingOptions('adam', ...
    'MaxEpochs', 1000, ...
    'InitialLearnRate', 0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 750, ...
    'Shuffle', 'every-epoch', ...
    'ValidationPatience', Inf, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% 训练模型
net = trainNetwork(P_train_cell, t_train, layers, options);


%% 预测
YPred = classify(net, P_test);

%% 性能评估
% 将categorical类型的预测和真实标签转换为数值型，以便计算混淆矩阵
predictedLabels = double(YPred);
trueLabels = double(t_test);

% 计算混淆矩阵
confMat = confusionmat(trueLabels, predictedLabels);

% 计算性能指标
numClasses = size(confMat, 1);
accuracy = sum(diag(confMat)) / sum(confMat(:));
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
F1 = zeros(numClasses, 1);

for i = 1:numClasses
    precision(i) = confMat(i, i) / sum(confMat(:, i));
    recall(i) = confMat(i, i) / sum(confMat(i, :));
    F1(i) = 2 * ((precision(i) * recall(i)) / (precision(i) + recall(i)));
end

% 显示结果
fprintf('混淆矩阵:\n');
disp(confMat);
fprintf('准确率: %.4f\n', accuracy);
fprintf('精确度:\n');
disp(precision);
fprintf('召回率:\n');
disp(recall);
fprintf('F1 分数:\n');
disp(F1);

toc;
