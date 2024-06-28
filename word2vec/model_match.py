import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import sys
import gensim
from gensim.models import KeyedVectors
import jieba
import jieba.analyse
import jieba.posseg as pseg
import pandas as pd
import numpy as np

#在模型基础上直接新增训练
def add_train(new_sentence, model):
    seg_list=jieba.cut(new_sentence, cut_all=False)
    sentences_cut=' '.join(seg_list)#分词
    model.build_vocab(sentences_cut, update=True)  # 注意update = True 这个参数很重要
    model.train(sentences_cut, total_examples=model.corpus_count, epochs=10)
    return model

#直接在词向量文件中添加
def add_vector(new_word,new_vector,file_path = "D:/word2vec/wiki.zh.text.vector"):
# 以追加模式打开文件
    with open(file_path, "a") as file:
        # 将新词及其向量格式化为字符串
        # 假设词向量是由空格分隔的浮点数
        vector_str = " ".join(map(str, new_vector))
        # 写入新词及其向量到文件末尾
        file.write(f"\n{new_word} {vector_str}")
    
    print("新词向量添加完成。")

#在python内部添加
def add_vector_inside(word, vector, kv):
    # 检查词是否已经存在
    if word in kv.key_to_index:
        print(f"Word '{word}' already exists in the model.")
        return
    
    # 向模型中添加新词向量
    kv.key_to_index[word] = len(kv.index_to_key)
    kv.index_to_key.append(word)
    
    # 添加向量。如果是添加第一个向量，需要特别处理
    if kv.vectors.shape[0] == 0:
        kv.vectors = np.array([vector])
    else:
        kv.vectors = np.vstack([kv.vectors, vector])
    
    # 更新norms（如果有）
    if hasattr(kv, 'vectors_norm'):
        kv.vectors_norm = np.vstack([kv.vectors_norm, vector / np.linalg.norm(vector)])



if __name__ == '__main__':
    fdir = 'D:/word2vec/'
    model = gensim.models.Word2Vec.load(fdir + 'wiki.zh.text.model')
    zh_model = KeyedVectors.load_word2vec_format(fdir + 'wiki.zh.text.vector')
    
    word_vectors=model.wv
    data_pro=pd.read_csv('D:/word2vec/固定投资.csv')
    data_1=pd.read_csv('D:/word2vec/1.csv')
    data_2=pd.read_csv('D:/word2vec/2.csv')
    data_3=pd.read_csv('D:/word2vec/3.csv')
    data=pd.concat([data_1,data_2],axis=0)
    data=pd.concat([data,data_3],axis=0)
    
    word_vec=gensim.wv.Word2Vec.load(fdir + 'wiki.zh.text.vector')
    
    seg_list=jieba.cut(data_pro.iloc[15307,2], cut_all=False)
    sent1=' '.join(seg_list)
    
    sent='云南中环有限公司年产50万吨瓦楞纸项目'
    tags_1 = jieba.analyse.extract_tags(sent, topK=4)#提取关键词
    
    seg_list=jieba.cut(data.iloc[114,-1], cut_all=False)
    sent=' '.join(seg_list)
    tags_2 = jieba.analyse.extract_tags(sent1, topK=3)
    similarity=word_vectors.n_similarity(['50'],['50'])
    
    thresh=0
    for i,pro in enumerate(data.iloc[:,-1].values):
        seg_list=jieba.cut(pro, cut_all=False)
        sent=' '.join(seg_list)
        similarity=word_vectors.n_similarity(sent1,sent)
        if similarity>thresh:
            s=sent
            thresh=similarity
    
    
    word_1 = model.wv.most_similar(u"50")
    word_2 = model.wv.most_similar(u"西财")
    for t in word_2:
        print(t[0],t[1])
    word_vectors['上财']
    #模型增量训练
    new_sentence = '上财就是上海财经大学'
    model=add_train(new_sentence,model)
    #add_vector('50.1',50.1*np.ones(400))
    add_vector_inside('50',50*np.ones(400),model.wv)
'''
    word = model.most_similar(positive=[u'皇上',u'国王'],negative=[u'皇后'])
    for t in word:
        print t[0],t[1]


    print model.doesnt_match(u'太后 妃子 贵人 贵妃 才人'.split())
    print model.similarity(u'书籍',u'书本')
    print model.similarity(u'逛街',u'书本')
'''
