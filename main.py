from model import BoostedBaggedTreeGAM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle

df = pd.read_excel('./CERML_origin.xlsx')
# Step 3: 'M' 열 제거
df = df.drop('M', axis=1)  # 'M'은 실제로 해당 열의 이름이어야 합니다.


# Step 4: 'G_Cl' 열을 예측 변수로 설정
y = df['G_Cl']  # 'G_Cl'은 실제로 해당 열의 이름이어야 합니다.
X = df.drop('G_Cl', axis=1)  # 'G_Cl'은 실제로 해당 열의 이름이어야 합니다.

column_names = X.columns.to_list()
column_dict = {i:name for i, name in enumerate(column_names)}

# 하이퍼 파라미터
k = 10
m_boostes = [1,2,3,4,5]
n_treeses = [10,30,50,70,100]   


# 결과 확인
print(X.head())
print(y.head())
X_np = X.values.astype('float32')
y_np = y.values.astype('float32').reshape(-1,1)

def k_fold_split(X, k):
    # 데이터의 총 길이를 가져옵니다.
    total_length = X.shape[0]

    # 데이터 인덱스를 생성합니다.
    indices = np.arange(total_length)

    # 인덱스를 무작위로 섞습니다.
    np.random.shuffle(indices)

    # 인덱스를 k 개의 그룹으로 분할합니다.
    split_indices = np.array_split(indices, k)

    # 각 fold에 대해 훈련 데이터와 검증 데이터의 인덱스를 생성합니다.
    folds = []
    for i in range(k):
        train_indices = np.concatenate([split_indices[j] for j in range(k) if j != i])
        val_indices = split_indices[i]
        folds.append((train_indices, val_indices))

    return folds


random_idx = np.random.permutation(X_np.shape[0])
X_np = X_np[random_idx]
y_np = y_np[random_idx]

folds = k_fold_split(X, k)



best_valid_error = 1000.
best_exp = None

for m_boost in m_boostes:
    for n_trees in n_treeses:
        
        exp_name = f"_K{k}M{m_boost}N{n_trees}"
        os.makedirs("./outputs"+exp_name,exist_ok=True)
        
        train_errors = np.zeros((k,),dtype=np.float32)
        valid_errors = np.zeros((k,),dtype=np.float32)
        feature_weights = np.zeros((k,X_np.shape[1]),dtype=np.float32)
        
        for i, (train_indices, val_indices) in enumerate(folds):
            print()
            print("---------------------------------------")
            print(f'Fold {i+1}')

            X_train = X_np[train_indices]
            y_train = y_np[train_indices]

            X_valid = X_np[val_indices]
            y_valid = y_np[val_indices]

            model = BoostedBaggedTreeGAM(m_boost=m_boost, n_leaves=2, n_trees=n_trees, pairwise=0)
            model.fit(X_train, y_train)

            print(f"{i} : train_error = ",np.square(model.predict(X_train) - y_train).mean())
            print(f"{i} : valid_error = ",np.square(model.predict(X_valid) - y_valid).mean())

            train_errors[i] = np.square(model.predict(X_train) - y_train).mean()
            valid_errors[i] = np.square(model.predict(X_valid) - y_valid).mean()
            
            feature_weights[i] = model.get_weights(X_train)

        print("feature_weights_mean =",feature_weights.mean(axis=0))
        feature_weights = feature_weights.mean(axis=0)

        plt.figure(figsize=(10,5))
        plt.bar(column_names, [float(f) for f in feature_weights], color='blue')

        # 축 이름과 제목 설정
        plt.xlabel('Names')
        plt.ylabel('Values')
        plt.title(exp_name + 'Bar Chart')

        # 그래프 저장
        plt.savefig("./outputs"+exp_name+'/feature_weights_bar.png')

        with open("./outputs"+exp_name+"/errors.json","w") as f:
            json.dump({"train_error" : float(train_errors.mean()),"valid_error" : float(valid_errors.mean())},f)

        with open("./outputs"+exp_name+"./feature_weights.json","w") as f:
            json.dump({column_names[i] : float(feature_weights[i]) for i in range(len(column_names))},f)
        
        plt.close()
        
        
        with open("./outputs"+exp_name+"./model.pickle","wb") as f:
            pickle.dump(model,f)
            
        
        
        if float(valid_errors.mean()) < best_valid_error:
            best_valid_error = valid_errors.mean()
            print("Best exp : ",exp_name)
            print("Best error : ",valid_errors.mean())
            with open("best_exp.txt","w") as f:
                f.write(exp_name + "\n")
                f.write(f"Best error : {float(valid_errors.mean())}")
            
            with open("Best_model.pickle","wb") as f:
                pickle.dump(model,f)









