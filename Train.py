import os
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
jump_path = os.path.join('pics', 'jump')    # 需要跳的图片的根目录
none_path = os.path.join('pics', 'none')    # 不需要跳的图片的根目录
# 所有图片的全路径
files = [os.path.join(jump_path, jump) for jump in os.listdir(jump_path)] + \
        [os.path.join(none_path, none) for none in os.listdir(none_path)]
X = []
y = [0] * len(os.listdir(jump_path)) + [1] * len(os.listdir(none_path))

# 遍历jump目录下的图片
for file in files:
    x = cv2.imread(file, 0).reshape(-1)
    X.append(x)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=1200, penalty='l2', solver='saga', tol=0.01)

model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(train_score, test_score)

# 保存模型
joblib.dump(model, 'dino_model.pth')