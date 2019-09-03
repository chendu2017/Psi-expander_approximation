# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:04:46 2018

@author: J-cd
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from gurobipy import gurobipy as grb
import math
import time
import numpy as np

#os.remove('results.txt')
def hoeffding_design_multiitem(cost,demand,a,num_plants,num_demand,service_level=0.1):
    
    fixed_cost = cost[0]
    holding_cost = cost[1]
    
    lower_demand = demand[0]
    mean_demand = demand[1]
    upper_demand = demand[2]
    
    items = range(lower_demand.shape[0])
    plants = range(num_plants)
    demand = range(num_demand)
    
    #记录每次加入的约束是Case1 or Case2 or Others0
    constraintType = []
    
    #记录每次迭代时候的objvalue
    iterValue = []
    SNum = []
    
    # ---- 主问题
    #申明主模型
    master_problem = grb.Model('master_problem') 
    master_problem.setParam('OutputFlag',0) 
    master_problem.modelSense = grb.GRB.MINIMIZE
    
    #申明决策变量 obj=***就表明了该组变量在目标函数中的系数
    ####这一块参考gurobi给的例子：facility.py
    I = master_problem.addVars(items,plants, vtype=grb.GRB.INTEGER, obj=holding_cost, name='I')
    
    Z = master_problem.addVars(plants, vtype=grb.GRB.BINARY, obj=fixed_cost, name='Z')
    
    # ---- 载入各种约束
    
    #利用hoeffding inequality计算总库存水平
    TI = np.zeros(shape=lower_demand.shape[0])
    for k in items:
        temp_sum = sum((upper_demand[k] - lower_demand[k])**2)
        TI[k] = math.ceil(mean_demand[k].sum() + math.sqrt(-math.log(service_level)*temp_sum/2))
    del temp_sum 
    
    
    #载入chance constraints 
    #至少有 demand个约束（至少每个需求点的需求，都能被和它相连的supplier满足吧）
    for k in items:
        for j in demand:
            expr = sum(a[:,j]*I.values()[13*k:13*(k+1)])
            master_problem.addQConstr(expr >= min(upper_demand[k,j], TI[k]-(lower_demand[k].sum()-lower_demand[k,j])))
            del expr
            #记录加入的约束的类型
            SNum.append(1)
            if upper_demand[k,j]<TI[k]-(lower_demand[k].sum()-lower_demand[k,j]):
                constraintType.append(1)
            if upper_demand[k,j]>TI[k]-(lower_demand[k].sum()-lower_demand[k,j]):
                constraintType.append(2)
            if upper_demand[k,j]==TI[k]-(lower_demand[k].sum()-lower_demand[k,j]):
                constraintType.append(3)
        
        
    
    #载入约束3-----Ii<=M*Zi
    M = upper_demand.sum()
    master_problem.addConstrs(I[k,i]<= M*Z[i] for i in plants)
    for i in plants: #记录加入的约束的类型
        constraintType.append(0)
        SNum.append(0)
    
    #total inventory的约束
    master_problem.addConstr(sum(I.values()[13*k:13*(k+1)]) == TI[k])
    constraintType.append(0)#记录加入的约束的类型
    SNum.append(0)
    
    # ---- 开始迭代主问题 
    add_constraints_time = 0
    
    time_start = time.clock()
    
    while True:    
    #解主问题
        master_problem.optimize()
    #确定主问题解的状态
        if  master_problem.status == 3:
            break
        
        elif  master_problem.status == 2:
            print('Master_Val:',master_problem.ObjVal)
            iterValue.append(master_problem.ObjVal)
        # ---- 开始子问题
            IterSubObjVal = np.zeros(shape=lower_demand.shape[0])
            IterX = np.zeros(shape=(lower_demand.shape[0],lower_demand.shape[1]))
            IterY = np.zeros(shape=(lower_demand.shape[0],lower_demand.shape[1]))
            IterZ = np.zeros(shape=(lower_demand.shape[0]))
            
            for k in items:
                #申明子问题    
                sub_problem = grb.Model('sub_problem') 
                sub_problem.setParam('OutputFlag',0)       
                
                #申明子问题的变量
                x = sub_problem.addVars(demand, vtype=grb.GRB.BINARY, name='x')
                y = sub_problem.addVars(plants, vtype=grb.GRB.BINARY, name='y')
                z = sub_problem.addVar(vtype=grb.GRB.CONTINUOUS, name='z')
                
                #设置目标函数
                expr1 = grb.quicksum([y[i] * I[k,i].x for i in plants])
                expr2 = grb.quicksum([upper_demand[k,j] * x[j] for j in demand])
                sub_problem.setObjective(expr1 - expr2 + z, grb.GRB.MINIMIZE)
                del expr1; del expr2 
                #载入子问题的约束
                #z>0约束
                sub_problem.addConstr(z >= 0)
            
                #子问题约束1
                expr1 = sum(upper_demand[k]*x.values())
                expr2 = grb.quicksum([lower_demand[k,j] * (1 - x[j]) for j in demand])
                sub_problem.addConstr(z >= expr1 - TI[k] + expr2)
                del expr1; del expr2
                     
                #子问题约束2
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem.addConstr(y[i] >= x[j])
                        
                #解子问题
                sub_problem.optimize()
                
                IterSubObjVal[k] = sub_problem.ObjVal
                for j in demand:
                    IterX[k,j] = x[j].x
                for i in plants:
                    IterY[k,i] = y[i].x
                IterZ[k] = z.x
                
            #判断子问题的目标函数值是否为非负数，若是，则主问题得到了最优解,break；否则加入新的约束到主问题中
            if all(IterSubObjVal>=-0.01):
                break
            else:
                
                for k in items:
                    
                    if IterSubObjVal[k] <= -0.001:
                        
                        expr1 = grb.quicksum([I[k,i]*IterY[k,i] for i in plants])
                        expr2 = sum([upper_demand[k,j]*IterX[k,j] for j in demand])
                        expr3 = sum([lower_demand[k,j]*(1-IterX[k,j]) for j in demand])
                        temp_expr = (expr1 >= min(expr2, TI[k]-expr3))
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        
                        #记录加入的约束的类型
                        if expr2<TI[k]-expr3:
                            constraintType.append(1)
                        if expr2>TI[k]-expr3:
                            constraintType.append(2)
                        if expr2==TI[k]-expr3:
                            constraintType.append(3)
                        SNum.append(sum([IterX[k,j] for j in demand]))
                            
                        del expr1;del expr2;del expr3;del temp_expr
                        add_constraints_time +=1
                
            
    time_end = time.clock()
    
    ################################到这里图就设计好了###################################
    ################################到这里图就设计好了###################################
    #输出一些图的参数(仓库数，道路数，总库存=TI)
    
    if  master_problem.status == 2:
    
        I_return = [[I[k,i].x for i in plants] for k in items] 
        Z_return = [Z[i].x for i in plants]
        open_DC = sum([Z[i].x for i in plants])
        p = TI
        
        return master_problem,constraintType,iterValue,SNum,['hoeffding',open_DC,I_return,Z_return,p,p/TI,master_problem.ObjVal,time_end-time_start,add_constraints_time,'optimal']
    
    elif master_problem.status == 3:   
        return master_problem,constraintType,iterValue,SNum,['-','-','-','-','-','-','-','-','-','infeasible']


def hoeffding_design(cost,demand,a,num_plants,num_demand,service_level=0.1):
    
    fixed_cost = cost[0]
    holding_cost = cost[1]
    
    lower_demand = demand[0]
    mean_demand = demand[1]
    upper_demand = demand[2]
    
    plants = range(num_plants)
    demand = range(num_demand)
    inventory = range(num_plants)
    
    #记录每次加入的约束是Case1 or Case2 or Others0
    constraintType = []
    
    #记录每次迭代时候的objvalue
    iterValue = []
    SNum = []
    
    #------------申明主问题-----------#
    #申明主模型
    master_problem = grb.Model('master_problem') 
    master_problem.setParam('OutputFlag',0) 
    master_problem.modelSense = grb.GRB.MINIMIZE
    
    #申明决策变量 obj=***就表明了该组变量在目标函数中的系数
    ####这一块参考gurobi给的例子：facility.py
    I = master_problem.addVars(inventory, vtype=grb.GRB.INTEGER, obj=holding_cost, name='I')
    
    Z = master_problem.addVars(plants, vtype=grb.GRB.BINARY, obj=fixed_cost, name='Z')
    
    #----------载入各种约束-----------#
    
    #利用hoeffding inequality计算总库存水平
    temp_sum = sum((upper_demand - lower_demand)**2)
    TI = math.ceil(mean_demand.sum() + math.sqrt(-math.log(service_level)*temp_sum/2))
    del temp_sum  
    
    
    #载入chance constraints 
    #至少有 demand个约束（至少每个需求点的需求，都能被和它相连的supplier满足吧）
    for j in demand:
        expr = sum(a[:,j]*I.values())
        master_problem.addQConstr(expr >= min(upper_demand[j], TI-(lower_demand.sum()-lower_demand[j])))
        del expr
        #记录加入的约束的类型
        SNum.append(1)
        if upper_demand[j]<TI-(lower_demand.sum()-lower_demand[j]):
            constraintType.append(1)
        if upper_demand[j]>TI-(lower_demand.sum()-lower_demand[j]):
            constraintType.append(2)
        if upper_demand[j]==TI-(lower_demand.sum()-lower_demand[j]):
            constraintType.append(3)
        
        
    
    #载入约束3-----Ii<=M*Zi
    M = upper_demand.sum()
    master_problem.addConstrs(I[i]<= M*Z[i] for i in plants)
    for i in plants: #记录加入的约束的类型
        constraintType.append(0)
        SNum.append(0)
    
    #total inventory的约束
    master_problem.addConstr(sum(I.values()) == TI)
    constraintType.append(0)#记录加入的约束的类型
    SNum.append(0)
    
    #------------------开始迭代主问题------------------------------------#
    #-------------------------------------------------------------------#
    add_constraints_time = 0
    
    time_start = time.clock()
    
    while True:    
    #解主问题
        master_problem.optimize()
    #确定主问题解的状态
        if  master_problem.status == 3:
            break
        
        elif  master_problem.status == 2:
            print('Master_Val:',master_problem.ObjVal)
            iterValue.append(master_problem.ObjVal)
    #-------------开始子问题---------------#
    #申明子问题    
            sub_problem = grb.Model('sub_problem') 
            sub_problem.setParam('OutputFlag',0)       
            
        #申明子问题的变量
            x = sub_problem.addVars(demand, vtype=grb.GRB.BINARY, name='x')
            y = sub_problem.addVars(plants, vtype=grb.GRB.BINARY, name='y')
            z = sub_problem.addVar(vtype=grb.GRB.CONTINUOUS, name='z')
            
        #设置目标函数
            expr1 = grb.quicksum([y[i] * I[i].x for i in plants])
            expr2 = grb.quicksum([upper_demand[j] * x[j] for j in demand])
            sub_problem.setObjective(expr1 - expr2 + z, grb.GRB.MINIMIZE)
            del expr1; del expr2 
        #载入子问题的约束
        #z>0约束
            sub_problem.addConstr(z >= 0)
        
        #子问题约束1
            expr1 = sum(upper_demand*x.values())
            expr2 = grb.quicksum([lower_demand[j] * (1 - x[j]) for j in demand])
            sub_problem.addConstr(z >= expr1 - TI + expr2)
            del expr1; del expr2
                 
        #子问题约束2
            for i in plants:
                for j in demand:
                    if round(a[i,j]) == 1:
                        sub_problem.addConstr(y[i] >= x[j])
                    
        #解子问题
            sub_problem.optimize()
        
    #判断子问题的目标函数值是否为非负数，若是，则主问题得到了最优解,break；否则加入新的约束到主问题中
            if sub_problem.ObjVal >= -0.001:
                break
            else:
                print(sub_problem.ObjVal)
                expr1 = grb.quicksum([I[i]*y[i].x for i in plants])
                expr2 = sum([upper_demand[j]*x[j].x for j in demand])
                expr3 = sum([lower_demand[j]*(1-x[j].x) for j in demand])
                temp_expr = (expr1 >= min(expr2, TI-expr3))
                master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                
                #记录加入的约束的类型
                if expr2<TI-expr3:
                    constraintType.append(1)
                if expr2>TI-expr3:
                    constraintType.append(2)
                if expr2==TI-expr3:
                    constraintType.append(3)
                SNum.append(sum([x[j].x for j in demand]))
                        
                
                del expr1;del expr2;del expr3;del temp_expr
                add_constraints_time +=1
            
    time_end = time.clock()
    
    ################################到这里图就设计好了###################################
    ################################到这里图就设计好了###################################
    #输出一些图的参数(仓库数，道路数，总库存=TI)
    
    if  master_problem.status == 2:
    
        I_return = []
        Z_return = []
        open_DC = 0
        for i in plants:
            I_return.append(I[i].x)
            Z_return.append(Z[i].x)
            open_DC = Z[i].x + open_DC
        p = TI
        
        return master_problem,constraintType,iterValue,SNum,['hoeffding',open_DC,I_return,Z_return,p,p/TI,master_problem.ObjVal,time_end-time_start,add_constraints_time,'optimal']
    
    elif master_problem.status == 3:   
        return master_problem,constraintType,iterValue,SNum,['-','-','-','-','-','-','-','-','-','infeasible']

