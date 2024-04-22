import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import time
import concurrent.futures
import asyncio
# 矩阵分解
start_time = time.time()
class MatrixFactorization():
    def __init__(self, sparse_m, K, alpha, beta, iterations):
        self.m = sparse_m#应该是初始输入的稀疏矩阵
        self.num_users, self.num_items = self.m.shape
        self.K = K#特征个数
        self.alpha = alpha#正则系数
        self.beta = beta#学习率
        self.iterations = iterations#迭代次数

    def train(self):
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))#正态分布
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))
        self.nonzero_indices = self.m.nonzero()#获取非0元素的索引，一般是一组元组包含行索引列表和列索引列表
        """self.data.nonzero()形式为(array(row),array(column))"""
        self.b = np.mean(self.m.data)#稀疏矩阵的data方法可以直接得到所有非零元的列表
        self.b_u = np.squeeze(rowwise_nonzero_mean(self.m.toarray()))-self.b#按行求非零元均值
        self.b_i = np.squeeze(colwise_nonzero_mean(self.m.toarray()))-self.b#按列求非零元均值
        
        #batch_size = 1024  # 可根据实际情况调整批量大小
        #futures = []
#
        #async def submit_and_collect_results(executor):
        #    nonlocal futures
        #    while futures:
        #        done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)
        #        futures = pending
        #        for future in done:
        #            future.result()  # 处理已完成任务的结果
#
        #loop = asyncio.get_event_loop()
        #executor = concurrent.futures.ProcessPoolExecutor()
#
        #for step in range(self.iterations):
        #    print(f"Iteration: {step}")
        #    for batch_start in range(0, len(self.nonzero_indices[0]), batch_size):
        #        batch_end = min(batch_start + batch_size, len(self.nonzero_indices[0]))
        #        batch_row_indices = self.nonzero_indices[0][batch_start:batch_end]
        #        batch_col_indices = self.nonzero_indices[1][batch_start:batch_end]
        #        batch_indices = list(zip(batch_row_indices, batch_col_indices))
        #        batch_tasks = [loop.run_in_executor(executor, self._update_factors, *indices)
        #                       for indices in batch_indices]
        #        futures.extend(batch_tasks)
#
        #    # 异步收集结果
        #    asyncio.ensure_future(submit_and_collect_results(executor))
#
        ## 等待所有任务完成
        #loop.run_until_complete(asyncio.gather(*futures))
        #loop.close()
        with concurrent.futures.ProcessPoolExecutor() as executor:#创建进程池
            futures=[]
            for step in range(self.iterations):#外层遍历迭代次数
                print(f"Iteration: {step}")
                for u, i in zip(*self.nonzero_indices):#内层封装异步任务
                    future = executor.submit(self._update_factors, u, i)#提交到进程池中
                    futures.append(future)
            #等待所有任务完成，，收集结果
            for future in concurrent.futures.as_completed(futures):
                future.result()#完成所有提交的任务，并行场景下无需关心任务完成的顺序
        #for step in range(self.iterations):
        #    print(f"Iteration: {step}")
        #    for u, i in zip(*self.nonzero_indices):#*号作用是把一个完整的元组，拆分为单独两个列表，zip可以进行两个列表按
        #原来序号的一一对应
        #        error = self.m.toarray()[u, i] - self.predict(u, i)
#
        #        self.b_u[u] += self.beta * (error - self.alpha * self.b_u[u])
        #        self.b_i[i] += self.beta * (error - self.alpha * self.b_i[i])
#
        #        self.P[u, :] += self.beta * (error * self.Q[i, :] - self.alpha * self.P[u, :])
        #        self.Q[i, :] += self.beta * (error * self.P[u, :] - self.alpha * self.Q[i, :])
    def _update_factors(self, u, i):
        error = self.m.toarray()[u, i] - self.predict(u, i)

        self.b_u[u] += self.beta * (error - self.alpha * self.b_u[u])
        self.b_i[i] += self.beta * (error - self.alpha * self.b_i[i])

        self.P[u, :] += self.beta * (error * self.Q[i, :] - self.alpha * self.P[u, :])
        self.Q[i, :] += self.beta * (error * self.P[u, :] - self.alpha * self.Q[i, :])

    def predict(self, u, i):#主要用于机器学习的训练循环和估计值预测中
        return self.b + self.b_u[u] + self.b_i[i] + np.dot(self.P[u, :], self.Q[i, :].T)

    def get_ratings(self):#直接给出预测矩阵，np.array类型
        pre_matrix=np.zeros((self.num_users,self.num_items))
        for i in range(self.num_users):
            for j in range(self.num_items):
                a=min(max(round(self.predict(i, j) / 0.5) * 0.5, 0.5), 5)#化为四舍五入的0.5的倍数
                pre_matrix[i,j]=a
        return pre_matrix

def rowwise_nonzero_mean(matrix):#按行计算矩阵非零元素的均值,这里的矩阵是由np.array生成的
    # 创建一个布尔掩码，标记出非零元素的位置
    mask = matrix !=0

    # 计算每行的非零元素个数
    nonzero_counts = np.sum(mask, axis=1)

    # 计算每行非零元素的和
    nonzero_sums = np.nansum(matrix, axis=1)

    # 计算并返回每行非零元素的均值
    return nonzero_sums / nonzero_counts

def colwise_nonzero_mean(matrix):#按列计算矩阵非零元素的均值,这里的矩阵是由np.array生成的
    # 使用 numpy 的 isnan 函数（或 isfinite，根据您的数据类型和需求调整）
    # 创建一个布尔掩码，标记出非零元素的位置
    mask = matrix !=0

    # 计算每列的非零元素个数
    nonzero_counts = np.sum(mask, axis=0)

    # 计算每列非零元素的和
    nonzero_sums = np.nansum(matrix, axis=0)

    # 计算并返回每行非零元素的均值
    return nonzero_sums / nonzero_counts

# 3. 评估
def eva_rmse(true_ratings, predicted_ratings):
    urm=true_ratings.toarray()
    uTest_rc_items=predicted_ratings
    sum,n=0,0
    for i in range(urm.shape[0]):
        for j in range(urm.shape[1]):
            if urm[i,j]>0:
                sum+=pow((urm[i,j]-uTest_rc_items[i,j]),2)
                n+=1
    RMSE=np.sqrt(sum/n)
    #rmse = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    return RMSE

#数据清洗,这一步非常非常非常重要
def load_data(dataset_url):# 数据收集
    data = pd.read_csv(dataset_url,encoding='gb18030')
    # 提取用户ID、电影ID和评分数据
    users = data['userId'].unique()#去重提取所有用户ID
    movies = data['movieId'].unique()#去重提取所有电影ID
    ratings = data[['userId', 'movieId', 'rating']].values#

    return ratings, users, movies

def preprocess_data(ratings, users, movies):#做出可以进行分解的稀疏矩阵
    # 将用户ID和电影ID转换为连续索引
    user_to_index = {user: index for index, user in enumerate(users)}#非常秀的操作
    """enumerate(users)可以给每个用户分配一个索引值"""
    movie_to_index = {movie: index for index, movie in enumerate(movies)}

    # 构建稀疏评分矩阵
    rows, cols = [], []
    for user_id, movie_id,rating in ratings:#映射
        rows.append(user_to_index[user_id])
        cols.append(movie_to_index[movie_id])

    rating_matrix = coo_matrix((ratings[:, 2], (rows, cols)), shape=(len(users), len(movies)))
    """ratings[:, 2]表示ratings的第三列"""
    return rating_matrix

if __name__ == '__main__':
    ratings,users,movies=load_data('C:/Users/13987/Desktop/澳科大学习实践/计算机体系结构/ml-25m/ratings.csv')
    rating_matrix=preprocess_data(ratings, users, movies)#初始的稀疏矩阵
    m=MatrixFactorization(rating_matrix,50,0.1,0.01,5)
    m.train()#训练出P和Q
    pre_matrix=m.get_ratings()
    RMSE=eva_rmse(rating_matrix,pre_matrix)
    end_time = time.time()
    duration = end_time - start_time
    print('Running:{:.2f} seconds'.format(duration))
    print('RMSE:{}'.format(RMSE))
