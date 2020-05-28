
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import json
import tensorflow as tf
import re
import string
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain
from collections import defaultdict
from collections import OrderedDict

from keras import backend as K
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

##################################################################################################

with open('Tr.json', encoding="utf-8-sig") as Tr_data_file:    
    Tr_data = json.load(Tr_data_file)
    
with open('Te.json', encoding="utf-8-sig") as Te_data_file:    
    Te_data = json.load(Te_data_file)
    
Tr_data.extend(Te_data)   
    
with open('Merge_data.json', 'w', encoding="utf-8") as new_file :
    json.dump(Tr_data, new_file, ensure_ascii = False, indent = "\t")
    
with open('Merge_data.json', encoding="utf-8") as new_file :
    Merge_data = json.load(new_file)

##################################################################################################
#  정규표현식을 사용하여 필요한 명사 추출
    
noun_list=[]
answer_list=[]

for i in range(len(Merge_data)) :
    
    word_list=[]
    for index in range(len(Merge_data[i].get('sentence'))):
        Merge_data[i].get('sentence')[index] = Merge_data[i].get('sentence')[index].replace("+" , " ")
        Merge_data[i].get('sentence')[index] = Merge_data[i].get('sentence')[index].strip()

        document_text = Merge_data[i].get('sentence')[index]
        match_pattern = re.findall(r'\b([가-힣]{1,10}\/NNG|[가-힣]{1,10}\/NNP)\b', document_text)
 
        for word in match_pattern:
            word = re.sub(r"\/NNG|\/NNP", "", word)
            word_list.append(word)
            
            
    noun_list.append(word_list)
    # "asnwer"을 Int로 바꾸는 작업
    str_to_int = ord(Merge_data[i].get('answer')[2]) - 49
    answer_list.append(str_to_int)
       
##################################################################################################
#  문서 전체의 추출 명사들을 Dictionary화 // 단어들을 Index와 Mapping (Encoding)
    
def word_to_num(list_2D):
    w2n_dic = dict()  # word가 key이고 index가 value인 dict
    n2w_dic = dict()  # index가 key이고 word가 value인 dict. 나중에 번호에서 단어로 쉽게 바꾸기 위해.
    idx = 1
    num_list = [[] for _ in range(len(list_2D))]   # 숫자에 매핑된 문서의 리스트
    
    for k,i in enumerate(list_2D):
        if not i:
            continue
            
        elif isinstance(i, str): 
            if w2n_dic.get(i) is None:
                w2n_dic[i] = idx
                n2w_dic[idx] = i
                idx += 1
                
            num_list[k] = [dic[i]]
            
        else:
            for j in i:
                if w2n_dic.get(j) is None:
                    w2n_dic[j] = idx
                    n2w_dic[idx] = j
                    idx += 1
                    
                num_list[k].append(w2n_dic[j])
                
    return num_list, w2n_dic, n2w_dic  
        
num_list, w2n_dic, n2w_dic = word_to_num(noun_list)

##################################################################################################
#  문서별 단어의 개수를 일정하게 맞춤(정규화) // 정답 데이터를 One-Hot 인코딩

vector_x = np.array(num_list)
vector_y = np.array(answer_list)
y_to_predict = vector_y

vector_x = sequence.pad_sequences(vector_x, maxlen = 50)  # 전체 문서 당 추출 명사 평균 개수 = 약 53.7개
vector_y = np_utils.to_categorical(vector_y)              # [0,0,0,1,0], [1,0,0,0,0] 과 같이 One-Hot Encoding


##################################################################################################
#   K-Fold 교차검증 및 모델링

# words_num은 총 단어의 종류. +1을 해준 이유는 단어 수가 적은 글의 경우 빈 칸에 0이 있기 때문에.
words_num = len(n2w_dic)
accuracy = []
i = 1

# K-Fold 교차검증 설정
kfold_obj = KFold(n_splits=5, shuffle=True, random_state = 0)

for train, validation in kfold_obj.split(vector_x, vector_y):
    
    model = Sequential()
    model.add(Embedding(words_num+1,len(vector_x[0])))  # (전체 추출 명사의 개수 = Input Size, 정규화한 단어의 개수 = Input length)
    #model.add(Dropout(0.3))
    #model.add(Conv1D(64, 5, padding='valid', activation='relu',strides=1))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.2))
    model.add(LSTM(len(vector_x[0])))
    model.add(Dense(len(vector_y[0]), activation='softmax'))  # 카테고리 수
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(vector_x[train], vector_y[train], batch_size=100, epochs=70,verbose =2, validation_data=(vector_x[validation], vector_y[validation]))
    kfold_accuracy = (model.evaluate(vector_x[validation], vector_y[validation])[1]*100)
    accuracy.append(kfold_accuracy)
    print("\n교차검증 %d/5 정확도 : %.2f%%\n\n" % (i, kfold_accuracy) )
    i=i+1

    
# 모델 구성 도식화 출력
model.summary()


##################################################################################################
#   결과물 출력
#    K-Fold 교차검증 정확도 및 평균 정확도 // 정확도 및 손실률 그래프 // 마지막 검증데이터 Precision, Recall, F1Score 출력


def mean(list):
    if len(list) == 0:
        return 0
    return sum(list) / len(list)

avr = mean(accuracy)

print('\n5-Fold Cross Validation 정확도 : {}'.format(accuracy))
print('\n5-Fold Cross Validation 평균 정확도 : %.2f%%' % avr)

y_loss = history.history['loss']
x_len = np.arange(len(y_loss))

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title("Loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(x_len, history.history['val_loss'], marker='.', c='red', label='val_set_loss')
plt.plot(x_len, history.history['loss'], marker='.', c='blue', label = 'train_set_loss')
plt.legend()
plt.grid()
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(x_len, history.history['val_acc'], marker='.', c='red', label='val_set_acc')
plt.plot(x_len, history.history['acc'], marker='.', c='blue', label = 'train_set_acc')
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()


y_true = y_to_predict[validation]
y_pred = model.predict_classes(vector_x[validation], batch_size=100, verbose=1)
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
print(classification_report(y_true, y_pred, target_names=target_names))

