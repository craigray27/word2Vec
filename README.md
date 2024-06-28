word2Vec includes 4 files: process.py and jieba.py are both used for pre-treatment of sentences downloaded from wiki. Due to the limited storage, the result of pretreatment (tremendous words, vacabularies) is strored in link：https://pan.quark.cn/s/eb41929568f7
wiki.zh.simp.word.txt, so directly run word2vec_model.py to get word vectors. The model is stored in wiki.zh.text.model and word vectors is stored in wiki.zh.text.vector.txt.

Bayesian_Neural_Network includes a Bayesian Neural Network based on the Variational Inference, and we can use word vectors generating above to do some new job like how close the two sentences are such that these tow features descrbing by these 2 sentences can link.
