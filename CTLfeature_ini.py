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
filename = "D:/candy/NB15/data/4_sum/ct-value/train.csv"
data = pd.read_csv(filename)

# 所有原始資料，將lable0,label1分開
label_0_data = data[data['label'] == 0]
label_1_data = data[data['label'] == 1]
print(f"原始資料0:{len(label_0_data)}")
print(f"原始資料1:{len(label_1_data)}")

# 所有越界資料
# 將總和改成平均
# label_0_data = label_0_data[label_0_data['sum']/(len(label_0_data.columns)-2)<=0]
# label_1_data = label_1_data[label_1_data['sum']/(len(label_0_data.columns)-2)>=0]
# print(f"越界資料0:{len(label_0_data)}")
# print(f"越界資料1:{len(label_1_data)}")

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

# 實際上可以當作絕對值相加
# 因為
# label_0_data_CTL 為負
# label_1_data_CTL 為正
# label_0_data_NCTL為負
# label_1_data_NCTL為正
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


# 欲輸出前X%的特徵数
drop_percent = 0.2
top_percent = int(len(sorted_sub) * drop_percent)

# 取出前X%的特徵名
#drop_columns = [item[0] for item in sorted_sub[:top_percent]]

#true_damage<0都丟棄
drop_columns = [
    feature for feature, value in sorted_sub
    if value < 0.1
]

'''
drop_columns = []

for column in label_0_data.columns.tolist():
    if result_dict['sub'][column] > 0.5:
        drop_columns.append(column)
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
sorted_sub_df.to_csv('sorted_sub.csv', index=False)
