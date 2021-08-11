from matplotlib import pyplot as plt
import pandas as pd
from glob import glob
import json

output = glob('./result/*.json')
dic = {}
for file_path in output:
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
        dic[json_data["result"][0]["percent"]] = json_data["result"][0]

df = pd.DataFrame(dic)
df = df.T
df = df.sort_values(by="percent",ascending=True)

#그래프 그리기
percent = df['percent'].tolist()
trainset_acc = df['trainset_acc'].tolist()
testset_acc = df['testset_acc'].tolist()
plt.plot(percent, trainset_acc, marker='o')
plt.plot(percent, testset_acc, marker='o')
plt.title('Weight pruning')
plt.xlabel('Pruning Percent')
plt.ylabel('Accuracy')
plt.legend(['Train set', 'Test set'])

#저장
plt.savefig('./result/graph/graph.png')
df.to_csv("./result/csv/result.csv", mode='w')
print("end")