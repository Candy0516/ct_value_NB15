'''
CTL=CrossTheLine
他會跑出兩張圖
第一張圖是資料裡面根據每個特徵中的
sub_CTL 就是越界資料(白為負,黑為正)相減
sub_NCTL 就是非越界資料(白為正，黑為負)相減
true_damage 是上面兩者相減

第二張圖將特徵依照true_damage排序

最後會輸出一個txt，是指定第二張圖的前幾%的特徵名稱
'''
import pandas as pd
import os
import matplotlib.pyplot as plt
# filename為跑完ct值的預設路徑
#filename = os.path.join('..', 'data', 'history', 'hospital_all', '4_sum', 'ct-value', 'train.csv')
#data = pd.read_csv(filename)
data = pd.read_csv('D:/candy/ct-value-master/dataset/sum/train.csv')
#test_data = pd.read_csv('D:/研究所/碩0/ct-value-master/ct-value-master/data/7_sum_test/benign_test.csv')
test_data = pd.read_csv('D:/candy/ct-value-master/knn_new3_sum.csv')

# 所有原始資料，將lable0,label1分開
label_0_data = data[data['label'] == 0]
label_1_data = data[data['label'] == 1]
print(f"原始資料0:{len(label_0_data)}")
print(f"原始資料1:{len(label_1_data)}")

# 所有越界資料
# 將總和改成平均
label_0_data_cross = label_0_data[label_0_data['sum']/(len(label_0_data.columns)-2)<=0]
label_1_data_cross = label_1_data[label_1_data['sum']/(len(label_0_data.columns)-2)>=0]
print(f"越界資料0:{len(label_0_data_cross)}")
print(f"越界資料1:{len(label_1_data_cross)}")

#畫出sum直方圖
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))

plt.hist(label_0_data_cross['sum']/(len(label_0_data.columns)-2), bins=30, alpha=0.5, label='Train', color='skyblue', edgecolor='black')

plt.xlabel('sum')
plt.ylabel('Frequency')
plt.title('Comparison of sum Distribution (Train)')
plt.legend()
plt.grid(True)
plt.show()

# 不需要label和sum
label_0_data = label_0_data.drop(labels = ['label','sum'],axis = 1)
label_1_data = label_1_data.drop(labels = ['label','sum'],axis = 1)

# 每個特徵各自計算CTL,NCTL
result_dict = {
    'label_0_data_CTL': {column: label_0_data[label_0_data[column]<=0][column].mean() for column in label_0_data.columns.tolist()},
    'label_1_data_CTL': {column: label_1_data[label_1_data[column]>0][column].mean() for column in label_1_data.columns.tolist()},
    'label_0_data_NCTL': {column: label_0_data[label_0_data[column]>0][column].mean() for column in label_0_data.columns.tolist()},
    'label_1_data_NCTL': {column: label_1_data[label_1_data[column]<=0][column].mean() for column in label_1_data.columns.tolist()}
}
#print(result_dict)

result_dict2 = {
    'label_0_data_CTL': {column: label_0_data[label_0_data[column]<=0][column].sum() for column in label_0_data.columns.tolist()},
    'label_1_data_CTL': {column: label_1_data[label_1_data[column]>0][column].sum() for column in label_1_data.columns.tolist()},
    'label_0_data_NCTL': {column: label_0_data[label_0_data[column]>0][column].sum() for column in label_0_data.columns.tolist()},
    'label_1_data_NCTL': {column: label_1_data[label_1_data[column]<=0][column].sum() for column in label_1_data.columns.tolist()}
}

# 實際上可以當作絕對值相加
# 因為
# label_0_data_CTL 為負
# label_1_data_CTL 為正
# label_0_data_NCTL為正
# label_1_data_NCTL為負
dmg_dict = {
    'sub_CTL' :{column: (result_dict['label_1_data_CTL'][column] if not pd.isna(result_dict['label_1_data_CTL'][column]) else 0) - (result_dict['label_0_data_CTL'][column] if not pd.isna(result_dict['label_0_data_CTL'][column]) else 0) for column in result_dict['label_0_data_CTL']},
    'sub_NCTL' :{column: (result_dict['label_0_data_NCTL'][column] if not pd.isna(result_dict['label_0_data_NCTL'][column]) else 0) - (result_dict['label_1_data_NCTL'][column] if not pd.isna(result_dict['label_1_data_NCTL'][column]) else 0) for column in result_dict['label_0_data_NCTL']}
}

# 將NCTL-CTL為實際損傷值
# 非越界-越界
# 因此true_damage越小，該特徵越界情況越嚴重
dmg_dict['true_damage'] = {column: dmg_dict['sub_NCTL'][column]  - dmg_dict['sub_CTL'][column]  for column in dmg_dict['sub_NCTL']}


# 繪製出各項特徵的計算參數
labels = dmg_dict.keys()
mean_values = [list(data.values()) for data in dmg_dict.values()]
columns = list(dmg_dict['sub_CTL'].keys())
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = range(len(columns))
for i, label in enumerate(labels):
    ax.bar(
        [pos + width * i for pos in x],
        mean_values[i],
        width,
        label=label
    )
ax.set_xlabel('Columns')
ax.set_ylabel('CTL DMG')
ax.set_title('Distribution of CTL DMG by Feature')
ax.set_xticks([pos + width for pos in x])
ax.set_xticklabels(x)
ax.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

'''
# 繪製出以true_damage排序(小到大)的特徵排序圖
# 字典按值排序
sorted_sub = sorted(dmg_dict['true_damage'].items(), key=lambda x: x[1])

columns = [item[0] for item in sorted_sub]
values = [item[1] for item in sorted_sub]
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))
plt.bar(columns, values)
plt.xlabel('Column Name')
plt.ylabel('Value')
plt.title('Bar Chart of Sorted true_damage')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
'''

# label_0_data_CTL 為負
# label_0_data_NCTL為正
# label_0_data_NCTL + label_0_data_CTL 為負則代表越界情況嚴重(分錯的情況比分對還多)(因為label_0的ct值應該要為正)

dmg_dict['true_damage_label0'] = {
    column: ((result_dict2['label_0_data_NCTL'][column] if not pd.isna(result_dict['label_0_data_NCTL'][column]) else 0) +
            (result_dict2['label_0_data_CTL'][column] if not pd.isna(result_dict['label_0_data_CTL'][column]) else 0))/ len(label_0_data[column])
    for column in result_dict2['label_0_data_CTL']
}

# label_1_data_CTL 為正
# label_1_data_NCTL為負
# label_1_data_NCTL + label_1_data_CTL 為正則代表越界情況嚴重(分錯的情況比分對還多)(因為label_1的ct值應該要為負)
dmg_dict['true_damage_label1'] = {
    column: ((result_dict2['label_1_data_NCTL'][column] if not pd.isna(result_dict['label_1_data_NCTL'][column]) else 0) +
            (result_dict2['label_1_data_CTL'][column] if not pd.isna(result_dict['label_1_data_CTL'][column]) else 0))/ len(label_1_data[column])
    for column in result_dict2['label_1_data_CTL']
}

#print(dmg_dict)

def plot_sorted_true_damage(true_damage_dict, title):
    sorted_sub = sorted(true_damage_dict.items(), key=lambda x: x[1])

    columns = [item[0] for item in sorted_sub]
    values = [item[1] for item in sorted_sub]

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    plt.bar(columns, values)
    plt.xlabel('Column Name')
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
'''
# 繪製 混合版 true_damage
plot_sorted_true_damage(dmg_dict['true_damage'], 'Bar Chart of Sorted true_damage (Overall)')

# 繪製 label0 的 true_damage
plot_sorted_true_damage(dmg_dict['true_damage_label0'], 'Bar Chart of Sorted true_damage (Label 0 Only)')

# 繪製 label1 的 true_damage
plot_sorted_true_damage(dmg_dict['true_damage_label1'], 'Bar Chart of Sorted true_damage (Label 1 Only)')
'''

'''
# 欲輸出前X%的特徵数
drop_percent = 0.2
top_percent = int(len(sorted_sub) * drop_percent)

# 取出前X%的特徵名
drop_columns = [item[0] for item in sorted_sub[:top_percent]]'''


#此段註解
'''
drop_columns = []

for column in label_0_data.columns.tolist():
    if result_dict['sub'][column] > 0.5:
        drop_columns.append(column)
'''
#

'''
# 輸出結果，並保存於txt檔中
print(f"drop_columns:{drop_columns}")
print(f"捨去{drop_percent*100:.0f}%")
print(f"原始特徵數量:{len(label_0_data.columns.tolist())}")
print(f"捨去特徵數量:{len(drop_columns)}")
with open('drop_columns.txt', 'w') as file:
    for column in drop_columns:
        file.write(column + '\n')

# 將 sorted_sub 匯出為 CSV 檔案
sorted_sub_df = pd.DataFrame(sorted_sub, columns=['Feature', 'true_damage'])
sorted_sub_df.to_csv('sorted_sub.csv', index=False)'''


# ======= 這裡開始是「加上new_sum」的邏輯 =======

# 計算每個特徵的權重
def get_weight(true_damage):
    if true_damage < -0.1:
        return 0
    elif true_damage < 0.2:
        return 1
    elif true_damage < 0.3:
        return 2
    elif true_damage < 0.4:
        return 3   
    else:
        return 4

# 計算每個特徵的權重dict
weight_dict = {column: get_weight(dmg_dict['true_damage_label0'][column]) for column in dmg_dict['true_damage_label0']}
print(weight_dict)
total_weight = sum(weight_dict.values())
print(total_weight)
# 計算每一筆資料的new_sum
def calculate_new_sum(row):
    new_sum = 0
    for column in weight_dict:
        new_sum += row[column] * weight_dict[column]
    return new_sum

# 套用到原始data上
data['new_sum'] = data.apply(calculate_new_sum, axis=1)

# 檢查結果
print(data[['label', 'sum', 'new_sum']].head())

# 如果你要存檔可以加上：
data.to_csv('D:/candy/ct-value-master/train_new_sum.csv', index=False)

# 所有原始資料，將lable0,label1分開
label_0_data = data[data['label'] == 0]
label_1_data = data[data['label'] == 1]
print(f"原始資料0:{len(label_0_data)}")
print(f"原始資料1:{len(label_1_data)}")

# 所有越界資料
# 將總和改成平均
label_0_data = label_0_data[label_0_data['new_sum']/total_weight<=0]
label_1_data = label_1_data[label_1_data['new_sum']/total_weight>=0]
print(f"越界資料0:{len(label_0_data)}")
print(f"越界資料1:{len(label_1_data)}")


# ======= test =======
# 分成label0與label1
test_label_0_data = test_data[test_data['label'] == 0]
test_label_1_data = test_data[test_data['label'] == 1]
print(f"原始資料0:{len(test_label_0_data)}")
print(f"原始資料1:{len(test_label_1_data)}")

# 所有越界資料
# 將總和改成平均
test_label_0_data_cross = test_label_0_data[test_label_0_data['sum'] / (len(test_label_0_data.columns) - 2) <= 0]
test_label_1_data_cross = test_label_1_data[test_label_1_data['sum'] / (len(test_label_0_data.columns) - 2) >= 0]
print(f"越界資料0:{len(test_label_0_data_cross)}")
print(f"越界資料1:{len(test_label_1_data_cross)}")

#畫出sum直方圖
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))

plt.hist(test_label_0_data_cross['sum']/ (len(test_label_0_data.columns) - 2), bins=30, alpha=0.5, label='Test', color='lightcoral', edgecolor='black')

plt.xlabel('sum')
plt.ylabel('Frequency')
plt.title('Comparison of new_sum Distribution (Test)')
plt.legend()
plt.grid(True)
plt.show()

# 丟掉label與sum欄位
test_label_0_data = test_label_0_data.drop(labels=['label', 'sum'], axis=1)
test_label_1_data = test_label_1_data.drop(labels=['label', 'sum'], axis=1)

test_result_dict = {
    'label_0_data_CTL': {column: test_label_0_data[test_label_0_data[column] <= 0][column].mean() for column in test_label_0_data.columns.tolist()},
    'label_1_data_CTL': {column: test_label_1_data[test_label_1_data[column] > 0][column].mean() for column in test_label_1_data.columns.tolist()},
    'label_0_data_NCTL': {column: test_label_0_data[test_label_0_data[column] > 0][column].mean() for column in test_label_0_data.columns.tolist()},
    'label_1_data_NCTL': {column: test_label_1_data[test_label_1_data[column] <= 0][column].mean() for column in test_label_1_data.columns.tolist()}
}

print(test_data.shape)

def calculate_new_sum(row, weight_dict):
    new_sum = 0
    for column in weight_dict:
        if column in row:  # 避免column不見
            new_sum += row[column] * weight_dict[column]
    return new_sum

# 計算並加到原始的test_data
test_data['new_sum'] = test_data.apply(lambda row: calculate_new_sum(row, weight_dict), axis=1)

# 檢查結果
print(test_data[['label', 'sum', 'new_sum']].head())
print(test_data.shape)
# 如果你要存檔可以加上：
test_data.to_csv('D:/candy/ct-value-master/knn_new3_newsum.csv', index=False)

# 分成label0與label1
test_label_0_data = test_data[test_data['label'] == 0]
test_label_1_data = test_data[test_data['label'] == 1]
print(f"原始資料0:{len(test_label_0_data)}")
print(f"原始資料1:{len(test_label_1_data)}")

# 所有越界資料
# 將總和改成平均
test_label_0_data_cross = test_label_0_data[test_label_0_data['new_sum'] / total_weight <= 0]
test_label_1_data_cross = test_label_1_data[test_label_1_data['new_sum'] / total_weight >= 0]
print(f"越界資料0:{len(test_label_0_data_cross)}")
print(f"越界資料1:{len(test_label_1_data_cross)}")


#畫出new_sum直方圖
plt.style.use('ggplot')
plt.figure(figsize=(12, 6))

plt.hist(label_0_data['new_sum']/ total_weight, bins=30, alpha=0.5, label='Train', color='skyblue', edgecolor='black')
plt.hist(test_label_0_data_cross['new_sum']/ total_weight, bins=30, alpha=0.5, label='Test', color='lightcoral', edgecolor='black')

plt.xlabel('new_sum')
plt.ylabel('Frequency')
plt.title('Comparison of new_sum Distribution (Train vs Test)')
plt.legend()
plt.grid(True)
plt.show()

'''
drop_columns = [column for column, weight in weight_dict.items() if weight == 0]
print(f"權重為0的特徵數量: {len(drop_columns)}")
print(f"要捨去的特徵: {drop_columns}")

# 從train刪掉
data_filtered = data.drop(columns=drop_columns)

# 從test刪掉
test_data_filtered = test_data.drop(columns=drop_columns)

data_filtered.to_csv('D:/研究所/碩0/ct-value-master/ct-value-master/dataset/train_new_sum_fs.csv', index=False)
test_data_filtered.to_csv('D:/研究所/碩0/ct-value-master/ct-value-master/dataset/test_new_sum_fs.csv', index=False)

print("已儲存 train_new_sum_fs.csv 和 test_new_sum_fs.csv")
'''
