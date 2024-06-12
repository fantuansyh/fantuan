function Data=one_hot6()
% 清除环境
clear all;
close all;
clc;

% 读取FASTA文件
[a, seq] = fastaread('test_label_2.fasta');
M = length(seq);

% 预处理序列，计算编码后的最大长度
max_length = 0;
for j = 1:M
    N = length(seq{j}); % 直接使用序列的长度
    if (N-1)*20 > max_length % 减1因为我们会删除一个氨基酸
        max_length = (N-1)*20; % 更新最大长度
    end
end

% 初始化Data矩阵
Data = zeros(M, max_length);

% one-hot编码处理
for j = 1:M
    z = seq{j}; % 获取当前序列
    N = length(z); % 获取当前序列的长度
    x = round((N+1)/2); % 计算需要删除的中间位置，基于当前长度
    if N > 0 % 确保序列非空
        z(x) = ''; % 删除中间位置的氨基酸
    end
    N = length(z); % 更新序列长度
    encoded_seq = '';

    % 对当前序列进行one-hot编码
    for i = 1:N
        one_hot = ['0,', repmat('0,', 1, 19)]; % 为每种氨基酸生成编码
        aa_index = find('ACDEFGHIKLMNPQRSTVWY' == z(i));
        if ~isempty(aa_index)
            one_hot((aa_index-1)*2 + 1) = '1';
        end
        encoded_seq = [encoded_seq, one_hot]; % 连接编码
    end
    encoded_seq = encoded_seq(1:end-1); % 移除末尾逗号
    
    % 将字符串编码转换为数字数组
    splitted = strsplit(encoded_seq, ',');
    Data(j, 1:length(splitted)) = str2double(splitted); % 更新Data矩阵
end
end
