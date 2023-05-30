import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json; import os; import pickle;

def find_rank(dictionary, key):
    target_value = dictionary[key]
    sorted_values = sorted(dictionary.values(), reverse=True)
    rank = sorted_values.index(target_value) + 1
    return rank

def calculate_r_squared(y_true, y_pred):
    mean_y = np.mean(y_true)
    ssr = np.sum((y_pred - y_true) ** 2)
    sst = np.sum((y_true - mean_y) ** 2)
    r_squared = 1 - (ssr / sst)
    return r_squared

df = pd.read_excel('./CERML_origin.xlsx')
# Step 3: 'M' 열 제거
df = df.drop('M', axis=1)  # 'M'은 실제로 해당 열의 이름이어야 합니다.


# Step 4: 'G_Cl' 열을 예측 변수로 설정
y = df['G_Cl']  # 'G_Cl'은 실제로 해당 열의 이름이어야 합니다.
X = df.drop('G_Cl', axis=1)  # 'G_Cl'은 실제로 해당 열의 이름이어야 합니다.

column_names = X.columns.to_list()
column_dict = {i:name for i, name in enumerate(column_names)}

X_np = X.values.astype('float32')
y_np = y.values.astype('float32').reshape(-1,1)

X_min = X_np.min(axis=0)
X_max = X_np.max(axis=0)

feat_min_max = {column_dict[i] : min_max for i,min_max in enumerate(zip(X_min,X_max))}
print(feat_min_max[column_dict[0]])

with open("./best_exp.txt","r") as f:
    bext_exp = f.readline()

for dir_path, dir_name, file_name in os.walk(os.getcwd()):
    if not "outputs" in dir_path:
        continue

    with open(os.path.join(dir_path,"model.pickle"),"rb") as f:
        model = pickle.load(f)

    with open(os.path.join(dir_path,"feature_weights.json"),"r") as f:
        feature_weights = json.load(f)
        
    y_pred = model.predict(X_np)
    R2 = calculate_r_squared(y_np,y_pred)
    plt.plot(y_np,y_np)
    plt.scatter(y_np, y_pred,color='red')
    plt.title(f'$R^2$ = {R2:.2f}')
    plt.xlabel('E')
    plt.ylabel('E_pred')
    plt.savefig(os.path.join(dir_path,f'parity_curve.png'))
    plt.close()

    N = 100
    for feat_idx, feat_name in column_dict.items():
        continue
        min,max = feat_min_max[feat_name]
        x = np.zeros([N,len(column_dict)]).astype(np.float32)
        x_feat = np.linspace(min,max,N)
        x[:,feat_idx] = x_feat
        y = model.feature_contribution(x,feat_idx).reshape(-1)
        
        rank = find_rank(feature_weights,feat_name)
        weight = feature_weights[feat_name]
        
        plt.plot(x_feat,y)
        plt.title(f'{feat_name} Feature function, weight = {weight:.2f} | {rank} / {len(feature_weights)}')  # 그래프 제목
        plt.xlabel(f'{feat_name}')  # X축 레이블
        plt.ylabel(f'{feat_name} feature contribution')  # Y축 레이블
        plt.savefig(os.path.join(dir_path,f'R{rank}_{feat_name}_feature_function.png'))
        plt.close()
    
    
        