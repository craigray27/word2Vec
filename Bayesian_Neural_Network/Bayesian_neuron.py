
import numpy as np
import matplotlib.pyplot as plt
from tf_keras.activations import relu
from tf_keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow_probability as tfp

#matplotlib inline

def f(x, sigma):
    return 10 * np.sin(2 * np.pi * (x)) + np.random.randn(*x.shape) * sigma

class BNN_VI():
    def __init__(self, prior_sigma_1=1.5, prior_sigma_2=0.1, prior_pi=0.5):
    # 先验分布假设的各种参数
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi

# (w0_mu,w0_rho)是用来采样w0的高斯分布的参数，其他类似
        self.w0_mu, self.b0_mu, self.w0_rho, self.b0_rho = self.init_weights([1, 5])
        self.w1_mu, self.b1_mu, self.w1_rho, self.b1_rho = self.init_weights([5, 10])
        self.w2_mu, self.b2_mu, self.w2_rho, self.b2_rho = self.init_weights([10, 1])

# 把所有的mu和rho放在一起好管理，模型里可学习参数是mu和rho，不是w和b
        self.mu_list = [self.w0_mu, self.b0_mu, self.w1_mu, self.b1_mu, self.w2_mu, self.b2_mu]
        self.rho_list = [self.w0_rho, self.b0_rho, self.w1_rho, self.b1_rho, self.w2_rho, self.b2_rho]
        self.trainables = self.mu_list + self.rho_list
        self.optimizer = Adam(0.08)

    def init_weights(self, shape):
# 初始化可学习参数mu和rho们
        w_mu = tf.Variable(tf.random.truncated_normal(shape, mean=0., stddev=1.))#mu 高斯分布的期望
        b_mu = tf.Variable(tf.random.truncated_normal(shape[1:], mean=0., stddev=1.))
        w_rho = tf.Variable(tf.zeros(shape))#高斯分布的标准差
        b_rho = tf.Variable(tf.zeros(shape[1:]))
        return w_mu, b_mu, w_rho, b_rho

    def sample_w_b(self):
# 根据mu和rho们，采样得到w和b们
        self.w0 = self.w0_mu + tf.math.softplus(self.w0_rho) * tf.random.normal(self.w0_mu.shape)
        self.b0 = self.b0_mu + tf.math.softplus(self.b0_rho) * tf.random.normal(self.b0_mu.shape)
        self.w1 = self.w1_mu + tf.math.softplus(self.w1_rho) * tf.random.normal(self.w1_mu.shape)
        self.b1 = self.b1_mu + tf.math.softplus(self.b1_rho) * tf.random.normal(self.b1_mu.shape)
        self.w2 = self.w2_mu + tf.math.softplus(self.w2_rho) * tf.random.normal(self.w2_mu.shape)
        self.b2 = self.b2_mu + tf.math.softplus(self.b2_rho) * tf.random.normal(self.b2_mu.shape)
        self.w_b_list = [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2]

    def forward(self, X):
        self.sample_w_b()
    
    # 简单的3层神经网络结构
        x = relu(tf.matmul(X, self.w0) + self.b0)#w_T*X+b 神经网络,matmul:矩阵乘法
        x = relu(tf.matmul(x, self.w1) + self.b1)#第二层
        self.y_pred = tf.matmul(x, self.w2) + self.b2#输出层
        return self.y_pred
    
    #z的先验分布的假设，两个高斯分布的线性组合
    def prior(self, w):
        return self.prior_pi_1 * self.gaussian_pdf(w, 0.0, self.prior_sigma_1) \
    + self.prior_pi_2 * self.gaussian_pdf(w, 0.0, self.prior_sigma_2)  
    #
    
    def gaussian_pdf(self, x, mu, sigma):#高斯分布的概率*密度*
        return tfp.distributions.Normal(mu,sigma).prob(x)

    def get_loss(self, y_label):
        self.loss = []
        for (w_or_b, mu, rho) in zip(self.w_b_list, self.mu_list, self.rho_list):
                q_z_w = tf.math.log(self.gaussian_pdf(w_or_b, mu, tf.math.softplus(rho)) + 1E-30)#z 后验分布的逼近
                p_z = tf.math.log(self.prior(w_or_b) + 1E-30)#这些参数w or b, 就是z的值
                self.loss.append(tf.math.reduce_sum(q_z_w - p_z))
        #log(p(D|z))=log(p(y|x,z)),D=(x_i,y_i),所以值取y_label,均值取y_pred,然后从中采样而来
        #回归问题中 p(y|x,z)通常选高斯分布，分类问题则为伯努利分布
        p_d_theta = tf.math.reduce_sum(tf.math.log(self.gaussian_pdf(y_label, self.y_pred, 1.0) + 1E-30))
                #
        self.loss.append(-p_d_theta)
        return tf.reduce_sum(self.loss)#Elbo loss被拆成两个部分构成

    def train(self, X, y_label):
        loss_list = []
        for _ in range(2000):#2000个epoch
            #一个永久存在的梯度计算，非持久的 GradientTape 默认情况下只能用来计算一次梯度
            with tf.GradientTape(persistent=True) as tape:
            #
                self.forward(X)
                loss = self.get_loss(y_label)
                gradients = tape.gradient(loss, self.trainables)#自动更新trainables的值
                self.optimizer.apply_gradients(zip(gradients, self.trainables))
                loss_list.append(loss.numpy())#loss.numpy():tf variable 的值
            del tape#删除tape，释放内存
        return loss_list#每一个epoch的loss都会存进来

    def predict(self, X):
        return [self.forward(X) for _ in range(300)]#monte-carlo法，确定随机变量y的期望
        

if __name__=='__main__':
    num_of_samples = 64 # 样本数
    noise = 1.0 # 噪音规模
    
    X = np.linspace(-0.5, 0.5, num_of_samples).reshape(-1, 1)
    y_label = f(X, sigma=noise) # 样本的label，训练y_train
    y_truth = f(X, sigma=0.0) # 样本的真实值,真实分布
    
    '''
    plt.scatter(X, y_label, marker='+', label='Training data')
    plt.plot(X, y_truth, label='Ground Truth')
    plt.title('Noisy training data and ground truth')
    plt.legend();
    '''
    
    X = X.astype('float32')#必须这样，不然会报错
    y_label = y_label.astype('float32')
    
    model = BNN_VI()
    loss_list = model.train(X,y_label)
    plt.plot(np.log(loss_list))
    plt.grid()