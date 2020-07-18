import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import random
import xlrd
import re

# 读取文件数据
src_file = 2
if src_file == 1:
    book = xlrd.open_workbook('./附件1：数据集1-终稿.xlsx')
else:
    book = xlrd.open_workbook('./附件2：数据集2-终稿.xlsx')

sheet0 = book.sheets()[0]
print('total rows:',sheet0.nrows)

x_points = sheet0.col_values(colx=1, start_rowx=2,end_rowx=sheet0.nrows)
Ax_point = x_points[0]
Bx_point = x_points[-1]
y_points = sheet0.col_values(colx=2, start_rowx=2,end_rowx=sheet0.nrows)
Ay_point = y_points[0]
By_point = y_points[-1]
z_points = sheet0.col_values(colx=3, start_rowx=2,end_rowx=sheet0.nrows)
Az_point = z_points[0]
Bz_point = z_points[-1]
type_points = sheet0.col_values(colx=4, start_rowx=3,end_rowx=sheet0.nrows-1)
type_points.insert(0,2)
type_points.append(3)
bad_points = sheet0.col_values(colx=5, start_rowx=2,end_rowx=sheet0.nrows)

# 类型转换
for i in range(len(type_points)):
    type_points[i] = int(type_points[i])

for i in range(len(bad_points)):
    bad_points[i] = int(bad_points[i])
# 数据处理

# 解题参数
if src_file == 1:
    Sigma = 0.001
    theta = 30
    Alpha1 = 25
    Alpha2 = 15
    Beta1 = 20
    Beta2 = 25
else:
    Sigma = 0.001
    theta = 20
    Alpha1 = 20
    Alpha2 = 10
    Beta1 = 15
    Beta2 = 20

VFlag = 1
HFlag = 0
EndFlag = 3
M_V_Dist = max(Alpha1,Alpha2) / Sigma
M_H_Dist = max(Beta1,Beta2) / Sigma

# # 蚁群所需参数
# m = 50
# pAlpha=1 # Alpha 表征信息素重要程度的参数  
# pBeta=5 # Beta 表征启发式因子重要程度的参数  
# Rho=0.1 # Rho 信息素蒸发系数  
# NC_max=200 # 最大迭代次数  
# Q=100 # 信息素增加强度系数  

n = len(x_points)
all_points = np.array([x_points,y_points,z_points])
all_points = all_points.T
Distances = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        Distances[i][j] = Distances[j][i] = np.linalg.norm(all_points[i] - all_points[j])


class PathObject(object):
    def __init__(self):
        self.info_dict = {}
        self.path_list = []


def read_file(filepath):
    path_list = []
    path = []
    with open(filepath,'r') as rf:
        lines = rf.readlines()
        for str_line in lines:                            
            str_line = str_line.strip()
            if str_line == '':
                continue

            if str_line.find('[') == -1:
                if len(path) > 0:
                    show_path.path_list = path
                    path_list.append(show_path)                    
                
                path = []
                show_path = PathObject()            
                info_items = re.split(r'[;]',str_line)                
                for item in info_items:                    
                    infos = re.split(r'[:,=]',item)                    
                    show_path.info_dict[infos[0]] = infos[1].strip()
            else:
                str_line = str_line[1:-1].strip()
                str_ele = str_line.split(' ')                                
                pos = []
                for ele in str_ele:
                    if ele.strip() != '':
                        pos.append(float(ele.strip()))
                path.append(pos)

    return path_list

def calc_point_types(path_list):
    path_pnumber_list = []
    for path in path_list:
        point_type = []        
        for point in path.path_list:
            index = 0
            for point_a in all_points:
                if abs(point[0] - point_a[0]) < 0.001 and \
                    abs(point[1] - point_a[1]) < 0.001 and \
                        abs(point[2] - point_a[2]) < 0.001 :
                    point_type.append(index)
                    break
                index += 1
        path_pnumber_list.append(point_type)
    
    return path_pnumber_list


path_list = read_file('F:\\Learn_Will\\数学建模\\20190918\\2019年中国研究生数学建模竞赛F题\\code\\showPoint.txt')
path_pnumber_list = calc_point_types(path_list)

class Ant(object):
    # path中的元素为每个点的序号
    def __init__(self,path): 
        self.path = path
        self.p_l = 0.4
        self.length = self.calc_length(path)

    def calc_length(self,path_):
        length = 0
        for i in range(len(path_)-1):
            length += Distances[path_[i]][path_[i+1]]
        return length
    
    def path_value(self):
        bad_probas = []
        probas = 1
        # print('path value:','*'*20)
        for i in range(len(self.path)):            
            if bad_points[self.path[i]] == 1:
                if type_points[self.path[i]] == VFlag:
                    dist = min(Distances[self.path[i-1]][i],M_V_Dist)
                    bad_probas.append(1-0.2*dist/M_V_Dist)
                    probas *= (1-0.2*dist/M_V_Dist)
                    # print('probas:',1-0.2*Distances[self.path[i-1]][i]/M_V_Dist)

                elif type_points[self.path[i]] == HFlag:
                    dist = min(Distances[self.path[i-1]][i],M_H_Dist)                    
                    bad_probas.append(1-0.2*dist/M_H_Dist)
                    probas *= (1-0.2*dist/M_H_Dist)
                    # print('probas:',1-0.05*Distances[self.path[i-1]][i]/M_V_Dist)

        # print('*'*30)
        return (self.p_l*(self.length/3333)+(1-self.p_l)*len(self.path))        
        # return self.length

class ACO(object):
    def __init__(self, ant_num=100, maxIter=200, alpha=2, beta=1, gama=1, rho=0.3, Q=2):
        self.ants_num = ant_num   # 蚂蚁个数
        self.ant_num_min = 50     # 最小种群数量        
        self.maxIter = maxIter    # 蚁群最大迭代次数
        self.alpha = alpha        # 信息启发式因子
        self.beta = beta          # 期望启发式因子
        self.gama = gama          # 角度的指数
        self.rho = rho            # 信息素挥发速度
        self.sigma = 0.97         # rho自适应常数
        self.rho_min = 0.05       # rho最小取值
        self.Q = Q                # 信息素强度      
        self.point_num = len(x_points) # 城市个数   
        self.dist_mat = Distances  # 城市距离矩阵
        self.success_set = []      # 成功的路线
        self.path_count = 30       # 初始假定段数
        self.bad_points = bad_points # 坏点列表
        
        print('dist_mat shape:',self.dist_mat.shape)
        self.points = all_points   # 点的坐标
        self.point_type = type_points
        self.final_best_path = []  # 最优路径
        self.final_best_path_length = 0 # 最优路径的长度
        ###########################
        self.phero_mat = np.ones((self.point_num,self.point_num))
        self.eta_mat = 1/(self.dist_mat + np.eye(len(self.dist_mat)))
        self.remain_path = []
        for point in self.points:
            self.remain_path.append(np.linalg.norm(self.points[-1]-point))
        ############################
        self.ants_info = np.zeros((self.maxIter, self.ants_num))  # 记录每次迭代后所有蚂蚁的路径长度信息
        self.best_path = np.zeros(self.maxIter)                   # 记录每次迭代后整个蚁群的“历史”最短路径长度  
        ############################
        self.solve()              # 完成算法的迭代更新
        self.display()

    def update_phero_mat(self,delta):        
        self.phero_mat = (1-self.rho)*self.phero_mat + delta
        # self.phero_mat[self.phero_mat < 0.5] = 0.5
        # self.phero_mat[self.phero_mat > 2] = 1.2        
        self.auto_rho()
        # 判断浓度是否超过上限
    
    # 种群规模自适应
    def auto_ant_num(self,iter_num):
        m = self.ants_num - 20 * self.ant_num_min*iter_num/self.maxIter
        if m < self.ant_num_min:
            m = self.ant_num_min
        return m
    
    # 信息素自适应
    def auto_Q(self,iter_num):        
        if iter_num < self.maxIter/5:
            Q = self.Q
        elif iter_num < self.maxIter*3/5:
            Q = 3*self.Q
        elif iter_num < self.maxIter:
            Q = 5*self.Q
        return Q

    # 挥发因子自适应
    def auto_rho(self):
        self.rho = self.rho*self.sigma
        if self.rho < self.rho_min:
            self.rho = self.rho_min

    # 启发因子改进
    def auto_eta_mat(self,start,end,cur_path_num):
        if self.remain_path[end] <= 1e-6:
            end_to_B = 1
        else:
            end_to_B = self.remain_path[end]

        path_count = self.path_count-cur_path_num
        if path_count <= 0:
            path_count = 1
        eta = path_count/(self.dist_mat[start][end]*end_to_B)

        p_times = 1
        if bad_points[end] == 1:
            if type_points[end] == VFlag:
                p_times = 1-self.dist_mat[start][end]/M_V_Dist
            elif type_points[end] == HFlag:
                p_times = 1-self.dist_mat[start][end]/M_H_Dist
        return eta*p_times

    def solve(self):
        iter_num = 0
        b_firstIter = 0
        while iter_num < self.maxIter:
            print('++++++++++++++',iter_num,'++++++++++++++')
            delta_phero_mat = np.zeros((self.point_num,self.point_num)) # 初始化每次迭代后信息素矩阵的增量
            ###################################################################################
            b_firstAnt = 0          
            self.best_points = []
            
            for i in range(int(self.auto_ant_num(iter_num))): # 种群规模自适应
                print('------------------',i,'------------------')
                # 每只蚂蚁相关参数初始化
                point_index1 = 0 # 该题中初始点固定
                point_index2 = 0 # 下一个选择出来的点
                last_point = 0   # 上一个点
                ant_path = [0] # 蚂蚁走过的路径
                passed_point = [0]  # 路过的点
                no_passed_point = list(set(range(self.point_num))-set(passed_point)) # 未路过的点
                h_err = 0 # 水平误差
                v_err = 0 # 垂直误差
                j = 0 # 没有什么实际意义，就是看看选了几个点
                b_success = 0 # 搜索成功
                while True:  # 对剩余的点进行访问
                    print('***********',j,'***********')

                    #-------------------------------在剩下的点中取出符合条件的--------------------------------                    
                    next_point_list = []                    
                    for k in range(self.point_num-len(passed_point)):
                        # 误差要小于限定值        
                        if self.point_type[no_passed_point[k]] == HFlag: # 水平校正点
                            if self.dist_mat[point_index1][no_passed_point[k]] * Sigma + v_err < Beta1 and \
                                self.dist_mat[point_index1][no_passed_point[k]] * Sigma + h_err < Beta2 and \
                                    self.points[point_index1][0] <= self.points[no_passed_point[k]][0] :  # 向前选择
                                next_point_list.append(no_passed_point[k])
                        elif self.point_type[no_passed_point[k]] == VFlag: # 垂直校正点
                            if self.dist_mat[point_index1][no_passed_point[k]] * Sigma + v_err < Alpha1 and \
                                self.dist_mat[point_index1][no_passed_point[k]] * Sigma + h_err < Alpha2 and \
                                    self.points[point_index1][0] <= self.points[no_passed_point[k]][0] :  # 向前选择
                                next_point_list.append(no_passed_point[k])
                        elif self.point_type[no_passed_point[k]] == EndFlag: # 终点
                            if self.dist_mat[point_index1][no_passed_point[k]] * Sigma + v_err < theta and \
                                self.dist_mat[point_index1][no_passed_point[k]] * Sigma + h_err < theta and \
                                    self.points[point_index1][0] <= self.points[no_passed_point[k]][0] :  # 向前选择
                                next_point_list.append(no_passed_point[k])

                    # print('可选择的点:',next_point_list)
                    if len(next_point_list) == 0:
                        print('没有选择下一个合适的点')
                        b_success = 0
                        break
                    next_point_list = np.array(next_point_list)
                    #-------------------------------------------------------------------------------------

                    #-------------------------------状态迁移矩阵计算----------------------------------------
                    up_proba = np.zeros(len(next_point_list))  # 初始化状态迁移概率的分子                     
                    for k in range(len(next_point_list)):
                        # 计算角度
                        if j < 1:
                            angle = np.pi                            
                        else:
                            cosangel = sum((self.points[last_point]-self.points[point_index1])*(self.points[next_point_list[k]]-self.points[point_index1]))/  \
                                (self.dist_mat[last_point][point_index1]*self.dist_mat[point_index1][next_point_list[k]])
                            angle = np.arccos(cosangel)

                        up_proba[k] = np.power(self.phero_mat[point_index1][next_point_list[k]],self.alpha) *  \
                            np.power(self.auto_eta_mat(point_index1,next_point_list[k],j),self.beta)* \
                                np.power(float(angle/np.pi),self.gama)
                    
                    proba = up_proba / sum(up_proba)                    
                    #-------------------------------------------------------------------------------------
                    
                    #-------------------------------改进轮盘赌---------------------------------------------
                    random_num = np.random.rand()
                    print('随机概率:',random_num)
                    print('proba',proba)
                    # 随机概率大于0.4直接使用最大值，提高最大的使用概率                    
                    if random_num > 1.0: # 0.4
                        index_need = np.where(proba == np.max(proba))[0] 
                        print(index_need)
                        point_index2 = next_point_list[index_need[0]]                        
                    else:
                        lun_pan = 0
                        for count in range(len(proba)):
                            lun_pan += proba[count]                            
                            if random_num < lun_pan:
                                point_index2 = next_point_list[count]
                                break
                            
                    print('蚂蚁:',i,' 第',j,'个点:',point_index2)
                    #-------------------------------------------------------------------------------------

                    #-------------------------------选出点后的处理-----------------------------------------
                    ant_path.append(point_index2)       # 将选出来的点加入路径
                    passed_point.append(point_index2)     # 标记已经访问过的路径                                   
                    index = no_passed_point.index(point_index2)
                    del no_passed_point[index]    # 从未访问节点中删除
                    
                    v_err += self.dist_mat[point_index1][point_index2]*Sigma  # 计算垂直累计误差
                    h_err += self.dist_mat[point_index1][point_index2]*Sigma  # 计算水平累计误差

                    last_point = point_index1
                    point_index1 = point_index2     # 保存所选择的点        
                    random_num = np.random.rand()            
                    if int(self.point_type[point_index2]) == 0:
                        if self.bad_points[point_index2] == 1:
                            if random_num < 0.2:
                                h_err = min(h_err,5)
                                # h_err = 0
                            else:
                                h_err = 0
                        else:
                            h_err = 0
                    elif int(self.point_type[point_index2]) == 1:
                        if self.bad_points[point_index2] == 1:
                            if random_num < 0.2:
                                v_err = min(v_err,5)
                                # v_err = 0
                            else:
                                v_err = 0
                        else:
                            v_err = 0
                    elif int(self.point_type[point_index2]) == 3: # 到达目的地
                        # 对到达目的地的点的误差还需要判断，使其小于指定的值
                        b_success = 1
                        break

                    print('v_err:',v_err,' h_err:',h_err)
                    j += 1  # 可以用来计算此时有多少个点
                    #-------------------------------------------------------------------------------------
                
                #-------------------------------到达目的地或者死亡------------------------------------
                if b_success == 0:
                    # 死亡应该要有相应的处理方法，这里是抛弃死亡的蚂蚁
                    print('该蚂蚁死亡')
                    continue                    
                print('该蚂蚁成功到达')
                self.path_count = j-1 # 蚂蚁成功到达时，选了几条路径

                c_path_length = Ant(ant_path).path_value()
                print('c_path_length:',c_path_length)
                if b_firstAnt == 0:
                    b_firstAnt = 1
                    self.best_points = ant_path
                    self.best_path[iter_num] = c_path_length
                else:
                    if c_path_length < self.best_path[iter_num]:
                        self.best_points = ant_path
                        self.best_path[iter_num] = c_path_length                        
                #-------------------------------------------------------------------------------------

            
            #--------------------------------------更新最短路径------------------------------------
            # 保存这一次迭代的最优解
            self.success_set.append(self.best_points)

            if b_firstIter == 0:
                if len(self.best_points) > 0:
                    b_firstIter = 1
                    self.final_best_path = self.best_points
                    self.final_best_path_length = self.best_path[iter_num]
            elif len(self.best_points) > 0 and \
                self.final_best_path_length > self.best_path[iter_num]:
                self.final_best_path = self.best_points
                self.final_best_path_length = self.best_path[iter_num]
            #-------------------------------------------------------------------------------------

            #--------------------------------------计算信息素--------------------------------------
            Q = self.auto_Q(iter_num)
            for l in range(len(self.best_points) - 1):
                # 这里信息素表示的是边，用哪种更合适？
                delta_phero_mat[self.final_best_path[l]][self.final_best_path[l+1]] = (Q / self.final_best_path_length) # 信息素公式
                # delta_phero_mat[self.best_points[l]][self.best_points[l+1]] = (Q / self.best_path[iter_num]) # 信息素公式
            self.update_phero_mat(delta_phero_mat)
            #-------------------------------------------------------------------------------------

            print('++++++++++++++',iter_num,'++++++++++++++')
            iter_num += 1
        print('#########################################')
            
    def show_table(self, path):
        plt.figure()
        ax = plt.axes(projection='3d')                
        point_list = []        
        for i in range(len(path)):
            point_list.append(self.points[path[i]])   

        result_point = np.array(point_list)                

        nx = np.array(x_points)
        ny = np.array(y_points)
        nz = np.array(z_points)
        ntype = np.array(type_points)
        hx = nx[ntype==0]
        hy = ny[ntype==0]
        hz = nz[ntype==0]
        vx = nx[ntype==1]
        vy = ny[ntype==1]
        vz = nz[ntype==1]

        ax.scatter(hx,hy,hz,color='yellow')
        ax.scatter(vx,vy,vz,color='blue')
        ax.scatter(Ax_point,Ay_point,Az_point,color='red')
        ax.scatter(Bx_point,By_point,Bz_point,color='red')
        ax.plot(result_point[:,0],result_point[:,1],result_point[:,2],color='green') # 点之间的连线

        for point_number in path:
            if self.bad_points[point_number] == 1:
                ax.text(self.points[point_number][0],self.points[point_number][1],self.points[point_number][2],'Bad')

        plt.title('path_value:'+str(Ant(path).path_value())+'\nlength:'+str(Ant(path).length)+'\nnode:'+str(len(path)))        
        

    def display(self):
        print('成功路线个数:',len(self.success_set))
        print('最短路径节点数:',len(self.final_best_path))
        # 显示图像
        # if len(self.final_best_path) > 0:
        #     self.show_table(self.final_best_path)
        path_length_list = []
        path_value_list = []
        path_node_list = []
        for path in self.success_set:
            path_length_list.append(Ant(path).length)
            path_node_list.append(len(path))            
            path_value_list.append(Ant(path).path_value())
            with open('problem3.txt','a+') as fw:
                fw.write('path_value='+str(Ant(path).path_value())+';length:'+str(Ant(path).length)+';node:'+str(len(path))+'\r\n')
                for pos in path:
                    fw.write(str(self.points[pos])+'\r\n')
                    
        print('路径value:',path_value_list)
        np_data = np.array(path_length_list)
        plt.figure()
        plt.plot(np_data)
        plt.title('length')
        np_data = np.array(path_value_list)
        plt.figure()
        plt.plot(np_data)
        plt.title('value')
        plt.figure()
        plt.plot(path_node_list)
        plt.title('node number')

        self.sort_path_list(self.success_set)
        plt.show()

    def sort_path_list(self,path_list):
        value_list = []
        for path in path_list:
            value_list.append(Ant(path).path_value())
        
        dst_path_list = []
        for i in range(len(value_list)):
            for k in range(i+1,len(value_list)):
                if value_list[i] > value_list[k]:
                    temp = value_list[i]
                    value_list[i] = value_list[k]
                    value_list[k] = temp
            dst_path_list.append(path_list[i])
        
        possibility_list = []
        iterMax = 100
        
        max_poss = 0
        max_point = 0
        iter1 = 0
        for path in dst_path_list:
            success_items = 0
            failed_items = 0            
            for i in range(iterMax):
                v_err = 0
                h_err = 0
                b_success = 1
                for j in range(1,len(path)):
                    v_err += self.dist_mat[path[j]][path[j-1]]*Sigma
                    h_err += self.dist_mat[path[j]][path[j-1]]*Sigma
                    rand_num = np.random.rand()

                    if self.point_type[path[j]] == VFlag:
                        if v_err > Alpha1 or h_err > Alpha2: # 失败
                            failed_items += 1
                            b_success = 0
                            break
                        else: # 成功
                            if self.bad_points[path[j]] == 1: # 坏点
                                if rand_num < 0.2:
                                    v_err = min(v_err,5)
                                else:
                                    v_err = 0
                                    # v_err = min(v_err,5)
                            else:
                                v_err = 0
                    elif self.point_type[path[j]] == HFlag:
                        if v_err > Beta1 or h_err > Beta2: # 失败
                            failed_items += 1
                            b_success = 0
                            break
                        else: # 成功
                            if self.bad_points[path[j]] == 1: # 坏点
                                if rand_num < 0.2:
                                    h_err = min(h_err,5)
                                else:                                 
                                    h_err = 0
                                    # h_err = min(h_err,5)
                            else:
                                h_err = 0
                if b_success == 1:
                    success_items += 1            
            if max_poss < 1-failed_items/iterMax:
                max_poss = 1-failed_items/iterMax
                max_point = iter1
            possibility_list.append(1-failed_items/iterMax)
            iter1 += 1
        print('max:',max_point,';',max_poss)
        with open('max.txt','w') as fw:
            fw.write(str(dst_path_list[max_point]))
        
        plt.figure()                
        plt.bar(list(range(len(possibility_list))),possibility_list)
        print(possibility_list)

ACO()
