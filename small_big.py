# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:20:22 2018

@author: chend
"""

from gurobipy import gurobipy as grb
import math
import time
import itertools
import numpy as np

def small_big_design(cost,demand,a,num_plants,num_demand,s,S,weight,service_level=0.1,mode='set'):
    
    if mode == 'set':
        add_constraints_time = 0
        
        fixed_cost = cost[0]
        holding_cost = cost[1]
        
        lower_demand = demand[0]
        mean_demand = demand[1]
        upper_demand = demand[2]
        
        inventory = range(num_plants)
        plants = range(num_plants)
        demand = range(num_demand)
        
        constraintType = []
        iterValue = []
        iterValue_small = []
        iterValue_big = []
    #------------申明主问题-----------#
        #申明主模型
        master_problem = grb.Model('master_problem') 
        master_problem.setParam('OutputFlag',0) 
        master_problem.modelSense = grb.GRB.MINIMIZE
        
        #申明决策变量 obj=***就表明了该组变量在目标函数中的系数
        ####这一块参考gurobi给的例子：facility.py
        I = master_problem.addVars(inventory, vtype=grb.GRB.CONTINUOUS, obj=holding_cost, name='I')
        
        Z = master_problem.addVars(plants, vtype=grb.GRB.BINARY, obj=fixed_cost, name='Z')
        
        #----------载入各种约束-----------#
        
        #额外算的值
        temp_sum = sum((upper_demand - lower_demand)**2)
        TI = math.ceil(mean_demand.sum() + math.sqrt(-math.log(service_level)*temp_sum/2))
        del temp_sum
             
        #载入chance constraints 
        #至少有 demand个约束（至少每个需求点的需求，都能被和它相连的supplier满足吧）
        #这里是small set的约束
        for j in demand:
            expr = sum(a[:,j]*I.values())
            master_problem.addConstr(expr >= upper_demand[j])
            del expr
            constraintType.append('small')
        
        #这里是big set的约束
        #要求|S|>25 （某数），最直观的的是取30时候，此时不属于这个集合的点是空集
        #所以没有约束。
        #第二个的约束去子问题中迭代加入
        expr = 0 
        del expr 
        
        #载入约束3-----Ii<=M*Zi
        M = upper_demand.sum()
        master_problem.addConstrs(I[i]<= M*Z[i] for i in plants)
        for i in plants:
            constraintType.append('others')
        #------------------开始迭代主问题------------------------------------#
        #-------------------------------------------------------------------#
        time_start = time.clock()
        
        while True:    
        #解主问题
            master_problem.optimize()
            
        #确定主问题解的状态
            if  master_problem.status == 3:
                iterValue.append(master_problem.ObjVal)
                break
            
            elif  master_problem.status == 2:
                print('master_Value:',master_problem.ObjVal)
                iterValue.append(master_problem.ObjVal)

            #-------------开始子问题---------------#
            #申明small set的子问题----sub_problem1  
                sub_problem1 = grb.Model('sub_problem1') 
                sub_problem1.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x1 = sub_problem1.addVars(demand, vtype=grb.GRB.BINARY, name='x1')
                y1 = sub_problem1.addVars(plants, vtype=grb.GRB.BINARY, name='y1')
                
            #设置目标函数
                expr1 = grb.quicksum([y1[i]*I[i].x for i in plants])
                expr2 = sum(upper_demand*x1.values())
                #for j in demand:
                #    expr2 = expr2 + upper_demand[j] * x1[j]
                sub_problem1.setObjective(expr1 - expr2, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                expr = sum(x1.values())
                sub_problem1.addConstr(expr <= s)
                del expr
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem1.addConstr(y1[i] >= x1[j])
                   
            #解子问题
                sub_problem1.optimize()
            
            #申明big set的子问题----sub_problem2  -----------
            #big set
            #big set
            #big set
                sub_problem2 = grb.Model('sub_problem1') 
                sub_problem2.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x2 = sub_problem2.addVars(demand, vtype=grb.GRB.BINARY, name='x2')
                y2 = sub_problem2.addVars(plants, vtype=grb.GRB.BINARY, name='y2')
                
            #设置目标函数
                expr1 = grb.quicksum([(1-y2[i])*I[i].x for i in plants])
                expr2 = grb.quicksum([lower_demand[j]*(1-x2[j]) for j in demand])
                sub_problem2.setObjective(expr2 - expr1, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                expr = grb.quicksum(x2.values())
                sub_problem2.addConstr(expr >= S)
                del expr
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem2.addConstr(y2[i] >= x2[j])
                   
            #解子问题
                sub_problem2.optimize()
            
            #判断子问题的目标函数值是否为非负数，若是，则主问题得到了最优解,break；否则加入新的约束到主问题中
                if sub_problem1.ObjVal >= -0.01 and sub_problem2.ObjVal >= -0.01:
                    iterValue_small.append(sub_problem1.ObjVal)
                    iterValue_big.append(sub_problem2.ObjVal)
                    break
                else:
                    if sub_problem1.ObjVal < -0.001:
                        print('small_set:',sub_problem1.ObjVal)
                        iterValue_small.append(sub_problem1.ObjVal)
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = x1[j].x * a[:,j] + a_temp
                            expr2 = upper_demand[j]*x1[j].x + expr2
                        for i in plants:
                            if round(a_temp[i])!=0:
                                expr1 = expr1 + round(a_temp[i])/round(a_temp[i]) * I[i] 
                        temp_expr = expr1 >= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        constraintType.append('small')
                        del expr1;del expr2;del temp_expr;del a_temp
                        add_constraints_time += 1
                    else:
                        iterValue_small.append(np.nan)
                    
                    if sub_problem2.ObjVal < -0.001:
                        print('big_set:',sub_problem2.ObjVal)
                        iterValue_big.append(sub_problem2.ObjVal)
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = x2[j].x *a[:,j] + a_temp
                            expr2 = lower_demand[j]*(int(not x2[j].x)) + expr2
                        for i in plants:
                            expr1 = expr1 + int(not a_temp[i])*I[i]             
                        temp_expr = expr1 <= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        constraintType.append('big')
                        del expr1;del expr2;del temp_expr; del a_temp
                        add_constraints_time += 1
                    else:
                        iterValue_big.append(np.nan)
                        
        time_end = time.clock()
        
        if  master_problem.status == 2:
            
            I_return = []
            Z_return = []
            p = 0
            open_DC = 0
            for i in plants:
                open_DC = open_DC + Z[i].x
                p = p + I[i].x
                I_return.append(I[i].x)
                Z_return.append(Z[i].x)
                
            return master_problem,constraintType,iterValue,iterValue_small,iterValue_big,[s,open_DC,I_return,Z_return,p,p/TI,master_problem.ObjVal,time_end-time_start,add_constraints_time,'optimal'] 
        
        elif master_problem.status == 3:   
            return master_problem,constraintType,iterValue,iterValue_small,iterValue_big,['-','-','-','-','-','-','-','-','-','infeasible']
    
    if mode == 'weight':
        add_constraints_time = 0
        
        fixed_cost = cost[0]
        holding_cost = cost[1]
        
        lower_demand = demand[0]
        mean_demand = demand[1]
        upper_demand = demand[2]
        
        inventory = range(num_plants)
        plants = range(num_plants)
        demand = range(num_demand)
        
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
        
        #额外算的值
        temp_sum = 0
        for j in demand:
            temp_sum += (upper_demand[j] - lower_demand[j])*(upper_demand[j] - lower_demand[j])
            
        TI = math.ceil(mean_demand.sum() + math.sqrt(-math.log(service_level)*temp_sum/2))
        del temp_sum
             
        #载入chance constraints 
        
        #没有约束。
        #第二个的约束去子问题中迭代加入
        expr = 0 
        
        del expr 
        
        #载入约束3-----Ii<=M*Zi
        M = upper_demand.sum()
        master_problem.addConstrs(I[i]<= M*Z[i] for i in plants)
        
        #------------------开始迭代主问题------------------------------------#
        #-------------------------------------------------------------------#
        time_start = time.clock()
        
        while True:    
        #解主问题
            master_problem.optimize()
            
        #确定主问题解的状态
            if  master_problem.status == 3:
                break
            
            elif  master_problem.status == 2:
    
            #-------------开始子问题---------------#
            #申明small set的子问题----sub_problem1  
                sub_problem1 = grb.Model('sub_problem1') 
                sub_problem1.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x1 = sub_problem1.addVars(demand, vtype=grb.GRB.BINARY, name='x1')
                y1 = sub_problem1.addVars(plants, vtype=grb.GRB.BINARY, name='y1')
                
            #设置目标函数
                expr1 = 0; expr2 = 0
                for i in plants:
                    expr1 = expr1 + y1[i] * I[i].x
                for j in demand:
                    expr2 = expr2 + upper_demand[j] * x1[j]
                sub_problem1.setObjective(expr1 - expr2, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                sub_problem1.addConstr(grb.quicksum(x1[j]*upper_demand[j] for j in demand) <= weight*M)
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem1.addConstr(y1[i] >= x1[j])
                   
            #解子问题
                sub_problem1.optimize()
            
            #申明big set的子问题----sub_problem2  -----------
            #big set
            #big set
            #big set
                sub_problem2 = grb.Model('sub_problem1') 
                sub_problem2.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x2 = sub_problem2.addVars(demand, vtype=grb.GRB.BINARY, name='x2')
                y2 = sub_problem2.addVars(plants, vtype=grb.GRB.BINARY, name='y2')
                
            #设置目标函数
                expr1 = 0; expr2 = 0
                for i in plants:
                    expr1 = expr1 + (1-y2[i]) * I[i].x
                for j in demand:
                    expr2 = expr2 + lower_demand[j] * (1-x2[j])
                sub_problem2.setObjective(expr2 - expr1, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                sub_problem2.addConstr(grb.quicksum(x1[j]*lower_demand[j] for j in demand) >= (1-weight)*M)
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem2.addConstr(y2[i] >= x2[j])
                   
            #解子问题
                sub_problem2.optimize()
            
            #判断子问题的目标函数值是否为非负数，若是，则主问题得到了最优解,break；否则加入新的约束到主问题中
                if sub_problem1.ObjVal >= 0 and sub_problem2.ObjVal >= 0:
                    break
                else:
                    if sub_problem1.ObjVal < -0.0001:
                        print('small_set:',sub_problem1.ObjVal)
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = x1[j].x * a[:,j] + a_temp
                            expr2 = upper_demand[j]*x1[j].x + expr2
                        for i in plants:
                            if round(a_temp[i])!=0:
                                expr1 = expr1 + round(a_temp[i])/round(a_temp[i]) * I[i] 
                        temp_expr = expr1 >= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        del expr1;del expr2;del temp_expr;del a_temp
                        add_constraints_time += 1
                    
                    if sub_problem2.ObjVal < -0.0001:
                        print('big_set:',sub_problem2.ObjVal)
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = x2[j].x *a[:,j] + a_temp
                            expr2 = lower_demand[j]*(int(not x2[j].x)) + expr2
                        for i in plants:
                            expr1 = expr1 + int(not a_temp[i])*I[i]             
                        temp_expr = expr1 <= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        del expr1;del expr2;del temp_expr; del a_temp
                        add_constraints_time += 1
                    
        time_end = time.clock()
        
        if  master_problem.status == 2:
            
            I_return = []
            Z_return = []
            p = 0
            open_DC = 0
            for i in plants:
                open_DC = open_DC + Z[i].x
                p = p + I[i].x
                I_return.append(I[i].x)
                Z_return.append(Z[i].x)
                
            return s,open_DC,I_return,Z_return,p,p/TI,master_problem.ObjVal,time_end-time_start,add_constraints_time,'optimal'   
        
        elif master_problem.status == 3:   
            return '-','-','-','-','-','-','-','-','-','infeasible'
        
        
        
    if mode == 'both':
        add_constraints_time = 0
        
        fixed_cost = cost[0]
        holding_cost = cost[1]
        
        lower_demand = demand[0]
        mean_demand = demand[1]
        upper_demand = demand[2]
        
        inventory = range(num_plants)
        plants = range(num_plants)
        demand = range(num_demand)
        
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
        
        #额外算的值
        temp_sum = 0
        for j in demand:
            temp_sum += (upper_demand[j] - lower_demand[j])*(upper_demand[j] - lower_demand[j])
            
        TI = math.ceil(mean_demand.sum() + math.sqrt(-math.log(service_level)*temp_sum/2))
        del temp_sum
             
        #载入chance constraints 
        #至少有 demand个约束（至少每个需求点的需求，都能被和它相连的supplier满足吧）
        #这里是small set的约束
        for j in demand:
            expr = 0
            for i in plants:
                expr = a[i,j] * I[i] + expr
            master_problem.addQConstr(expr >= upper_demand[j])
            del expr
        
        #这里是big set的约束
        #要求|S|>25 （某数），最直观的的是取30时候，此时不属于这个集合的点是空集
        #所以没有约束。
        #第二个的约束去子问题中迭代加入
        expr = 0 
        
        del expr 
        
        #载入约束3-----Ii<=M*Zi
        M = upper_demand.sum()
        master_problem.addConstrs(I[i]<= M*Z[i] for i in plants)
        
        #------------------开始迭代主问题------------------------------------#
        #-------------------------------------------------------------------#
        time_start = time.clock()
        
        while True:    
        #解主问题
            master_problem.optimize()
            
        #确定主问题解的状态
            if  master_problem.status == 3:
                break
            
            elif  master_problem.status == 2:
    
            #-------------开始子问题---------------#
            
                '''
                2个和set大小有关的子问题
                '''
            #申明small set的子问题----sub_problem1  
                sub_problem1 = grb.Model('sub_problem1') 
                sub_problem1.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x1 = sub_problem1.addVars(demand, vtype=grb.GRB.BINARY, name='x1')
                y1 = sub_problem1.addVars(plants, vtype=grb.GRB.BINARY, name='y1')
                
            #设置目标函数
                expr1 = 0; expr2 = 0
                for i in plants:
                    expr1 = expr1 + y1[i] * I[i].x
                for j in demand:
                    expr2 = expr2 + upper_demand[j] * x1[j]
                sub_problem1.setObjective(expr1 - expr2, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                expr = 0 
                for j in demand:
                    expr = x1[j] + expr
                sub_problem1.addConstr(expr <= s)
                del expr
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem1.addConstr(y1[i] >= x1[j])
                   
            #解子问题
                sub_problem1.optimize()
            
            #申明big set的子问题----sub_problem2  -----------
            #big set
            #big set
            #big set
                sub_problem2 = grb.Model('sub_problem2') 
                sub_problem2.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x2 = sub_problem2.addVars(demand, vtype=grb.GRB.BINARY, name='x2')
                y2 = sub_problem2.addVars(plants, vtype=grb.GRB.BINARY, name='y2')
                
            #设置目标函数
                expr1 = 0; expr2 = 0
                for i in plants:
                    expr1 = expr1 + (1-y2[i]) * I[i].x
                for j in demand:
                    expr2 = expr2 + lower_demand[j] * (1-x2[j])
                sub_problem2.setObjective(expr2 - expr1, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                expr = 0 
                for j in demand:
                    expr = x2[j] + expr
                sub_problem2.addConstr(expr >= S)
                del expr
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem2.addConstr(y2[i] >= x2[j])
                   
            #解子问题
                sub_problem2.optimize()
                
                '''
                2个和weight 有关的子问题
                '''
               
            #申明small set的子问题----sub_problem1  
                sub_problem3 = grb.Model('sub_problem3') 
                sub_problem3.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x3 = sub_problem3.addVars(demand, vtype=grb.GRB.BINARY, name='x3')
                y3 = sub_problem3.addVars(plants, vtype=grb.GRB.BINARY, name='y3')
                
            #设置目标函数
                expr1 = 0; expr2 = 0
                for i in plants:
                    expr1 = expr1 + y3[i] * I[i].x
                for j in demand:
                    expr2 = expr2 + upper_demand[j] * x3[j]
                sub_problem3.setObjective(expr1 - expr2, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                sub_problem3.addConstr(grb.quicksum(x3[j]*upper_demand[j] for j in demand) <= weight*M)
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem3.addConstr(y3[i] >= x3[j])
                   
            #解子问题
                sub_problem3.optimize()
            
            #申明big weight的子问题----sub_problem2  -----------
            #big set
            #big set
            #big set
                sub_problem4 = grb.Model('sub_problem4') 
                sub_problem4.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x4 = sub_problem4.addVars(demand, vtype=grb.GRB.BINARY, name='x4')
                y4 = sub_problem4.addVars(plants, vtype=grb.GRB.BINARY, name='y4')
                
            #设置目标函数
                expr1 = 0; expr2 = 0
                for i in plants:
                    expr1 = expr1 + (1-y4[i]) * I[i].x
                for j in demand:
                    expr2 = expr2 + lower_demand[j] * (1-x4[j])
                sub_problem4.setObjective(expr2 - expr1, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                sub_problem4.addConstr(grb.quicksum(x4[j]*lower_demand[j] for j in demand) >= (1-weight)*M)
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem4.addConstr(y4[i] >= x4[j])
                   
            #解子问题
                sub_problem4.optimize()
            
            #判断子问题的目标函数值是否为非负数，若是，则主问题得到了最优解,break；否则加入新的约束到主问题中
                if sub_problem1.ObjVal >= 0 and sub_problem2.ObjVal >= 0 and sub_problem3.ObjVal >= 0 and sub_problem4.ObjVal >= 0:
                    break
                else:
                    if sub_problem1.ObjVal < -0.0001:
                        print('small_set:',sub_problem1.ObjVal)
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = x1[j].x * a[:,j] + a_temp
                            expr2 = upper_demand[j]*x1[j].x + expr2
                        for i in plants:
                            if round(a_temp[i])!=0:
                                expr1 = expr1 + round(a_temp[i])/round(a_temp[i]) * I[i] 
                        temp_expr = expr1 >= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        del expr1;del expr2;del temp_expr;del a_temp
                        add_constraints_time += 1
                    
                    if sub_problem2.ObjVal < -0.0001:
                        print('big_set:',sub_problem2.ObjVal)
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = x2[j].x *a[:,j] + a_temp
                            expr2 = lower_demand[j]*(int(not x2[j].x)) + expr2
                        for i in plants:
                            expr1 = expr1 + int(not a_temp[i])*I[i]             
                        temp_expr = expr1 <= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        del expr1;del expr2;del temp_expr; del a_temp
                        add_constraints_time += 1
                    
                    if sub_problem3.ObjVal < -0.0001:    
                        print('small_set:',sub_problem3.ObjVal)
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = x3[j].x * a[:,j] + a_temp
                            expr2 = upper_demand[j]*x3[j].x + expr2
                        for i in plants:
                            if round(a_temp[i])!=0:
                                expr1 = expr1 + round(a_temp[i])/round(a_temp[i]) * I[i] 
                        temp_expr = expr1 >= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        del expr1;del expr2;del temp_expr;del a_temp
                        add_constraints_time += 1
                        
                    if sub_problem4.ObjVal < -0.0001:
                        print('big_set:',sub_problem4.ObjVal)
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = x4[j].x *a[:,j] + a_temp
                            expr2 = lower_demand[j]*(int(not x4[j].x)) + expr2
                        for i in plants:
                            expr1 = expr1 + int(not a_temp[i])*I[i]             
                        temp_expr = expr1 <= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        del expr1;del expr2;del temp_expr; del a_temp
                        add_constraints_time += 1
                    
        time_end = time.clock()
        
        if  master_problem.status == 2:
            
            I_return = []
            Z_return = []
            p = 0
            open_DC = 0
            for i in plants:
                open_DC = open_DC + Z[i].x
                p = p + I[i].x
                I_return.append(I[i].x)
                Z_return.append(Z[i].x)
                
            return master_problem,constraintType,iterValue,[s,open_DC,I_return,Z_return,p,p/TI,master_problem.ObjVal,time_end-time_start,add_constraints_time,'optimal' ]  
        
        elif master_problem.status == 3:   
            return master_problem,constraintType,iterValue,['-','-','-','-','-','-','-','-','-','infeasible']
        
        
        
def small_big_fulldesign(cost,demand,a,num_plants,num_demand,s,S,weight,service_level=0.1):
    '''
    把exponential多的约束全部加入到模型中，解一个大模型
    '''
    fixed_cost = np.asarray(cost[0])
    holding_cost = np.asarray(cost[1])
    
    lower_demand = demand[0]
    mean_demand = demand[1]
    upper_demand = demand[2]
    
    inventory = range(num_plants)
    plants = range(num_plants)
    demand = range(num_demand)
    
    #------------申明主问题-----------#
    #申明主模型
    master_problem = grb.Model('master_problem') 
    master_problem.setParam('OutputFlag',0) 
    master_problem.modelSense = grb.GRB.MINIMIZE
    
    #申明决策变量 obj=***就表明了该组变量在目标函数中的系数
    ####这一块参考gurobi给的例子：facility.py
    I = master_problem.addVars(inventory, vtype=grb.GRB.CONTINUOUS, obj=holding_cost, name='I')
    
    Z = master_problem.addVars(plants, vtype=grb.GRB.BINARY, obj=fixed_cost, name='Z')
    
    #----------载入各种约束-----------#
    
    #额外算的值
    temp_sum = sum((upper_demand - lower_demand)**2)
    TI = math.ceil(mean_demand.sum() + math.sqrt(-math.log(service_level)*temp_sum/2))
    del temp_sum
         
    #载入chance constraints 
    #至少有 demand个约束（至少每个需求点的需求，都能被和它相连的supplier满足吧）
    #这里是small set的约束
    for k in range(1,s+1):
        for nodeSet in itertools.combinations(range(num_demand),k):            
            expr = sum( (a[:,nodeSet].sum(axis=1)>=1)*I.values())
            master_problem.addConstr(expr >= sum(upper_demand[j] for j in nodeSet))
            del expr
        print('Small=={},约束添加完成'.format(k))
    
    
    #这里是big set的约束
    
    for k in range(S+1,num_demand+1):
        for nodeSet in itertools.combinations(range(num_demand),k):
            
            nonNodeSet = [j for j in range(num_demand) if j not in nodeSet]
            nonAdjacentNodeSet = np.where(a[:,nodeSet].sum(axis=1)==0)[0]
            expr_I = sum(I[i] for i in nonAdjacentNodeSet)
            expr_DL = sum(lower_demand[j] for j in nonNodeSet)
            
            if nonAdjacentNodeSet.size != 0: #if nonAdjacentNodeSet.size==0, expr_i==0,下面的等式始终成立
                master_problem.addConstr(expr_I <= expr_DL)
            del expr_I,expr_DL
        
        print('Big=={},约束添加完成'.format(k))
    
    #载入约束3-----Ii<=M*Zi
    M = upper_demand.sum()
    master_problem.addConstrs(I[i]<= M*Z[i] for i in plants)
    
    #求解
    master_problem.optimize()
    
    return master_problem,{'# of Open DC': sum(Z.values()),
                           'FSC': sum(Z.values()*fixed_cost)+sum(I.values()*holding_cost),
                           'Inventory': I.values(),
                           'Total Inventory': sum(I.values())}
    
    

def small_big_design_multiitem(cost,demand,a,num_plants,num_demand,s,S,weight,service_level=0.1,mode='set'):
    
    add_constraints_time = 0
    
    fixed_cost = cost[0]
    holding_cost = cost[1]
    
    lower_demand = demand[0]
    mean_demand = demand[1]
    upper_demand = demand[2]
    
    items = range(lower_demand.shape[0])
    plants = range(num_plants)
    demand = range(num_demand)
    
    constraintType = []
    iterValue = []
    iterValue_small = []
    iterValue_big = []
    
    # ---- 申明主问题
    #申明主模型
    master_problem = grb.Model('master_problem') 
    master_problem.setParam('OutputFlag',0) 
    master_problem.modelSense = grb.GRB.MINIMIZE
    
    #申明决策变量 obj=***就表明了该组变量在目标函数中的系数
    ####这一块参考gurobi给的例子：facility.py
    I = master_problem.addVars(items,plants, vtype=grb.GRB.CONTINUOUS, obj=holding_cost, name='I')
    
    Z = master_problem.addVars(plants, vtype=grb.GRB.BINARY, obj=fixed_cost, name='Z')
    
    # ---- 载入各种约束
    
    #额外算的值
    TI = np.zeros(shape=lower_demand.shape[0])
    for k in items:
        temp_sum = sum((upper_demand[k] - lower_demand[k])**2)
        TI[k] = math.ceil(mean_demand[k].sum() + math.sqrt(-math.log(service_level)*temp_sum/2))
    del temp_sum
         
    #载入chance constraints 
    #至少有 demand个约束（至少每个需求点的需求，都能被和它相连的supplier满足吧）
    #这里是small set的约束
    for k in items:
        for j in demand:
            expr = sum(a[:,j]*I.values()[13*k:13*(k+1)])
            master_problem.addConstr(expr >= upper_demand[k,j])
            del expr
            constraintType.append('small')
    
    #这里是big set的约束
    #要求|S|>25 （某数），最直观的的是取30时候，此时不属于这个集合的点是空集
    #所以没有约束。
    #第二个的约束去子问题中迭代加入
    expr = 0 
    del expr 
    
    #载入约束3-----Ii<=M*Zi
    M = upper_demand.sum()
    master_problem.addConstrs(I[k,i]<= M*Z[i] for k in items for i in plants)
    for i in plants:
        constraintType.append('others')
        
    # ---- 开始迭代主问题
    time_start = time.clock()
    
    while True:    
    #解主问题
        master_problem.optimize()
        
    #确定主问题解的状态
        if  master_problem.status == 3:
            iterValue.append(master_problem.ObjVal)
            break
        
        elif  master_problem.status == 2:
            print('master_Value:',master_problem.ObjVal)
            iterValue.append(master_problem.ObjVal)
        
            # ---- 开始子问题
            IterSubObjVal = np.zeros(shape=(lower_demand.shape[0],2))
            IterX = np.zeros(shape=(lower_demand.shape[0],lower_demand.shape[1],2))
            IterY = np.zeros(shape=(lower_demand.shape[0],lower_demand.shape[1],2))
            
            # K-th item:
            for k in items:
            
            #申明small set的子问题----sub_problem1  
                sub_problem1 = grb.Model('sub_problem1') 
                sub_problem1.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x1 = sub_problem1.addVars(demand, vtype=grb.GRB.BINARY, name='x1')
                y1 = sub_problem1.addVars(plants, vtype=grb.GRB.BINARY, name='y1')
                
            #设置目标函数
                expr1 = grb.quicksum([y1[i]*I[k,i].x for i in plants])
                expr2 = sum(upper_demand[k]*x1.values())
                #for j in demand:
                #    expr2 = expr2 + upper_demand[j] * x1[j]
                sub_problem1.setObjective(expr1 - expr2, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                expr = sum(x1.values())
                sub_problem1.addConstr(expr <= s)
                del expr
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem1.addConstr(y1[i] >= x1[j])
                   
            #解子问题
                sub_problem1.optimize()
            
            #申明big set的子问题----sub_problem2  -----------
            #big set
            #big set
            #big set
                sub_problem2 = grb.Model('sub_problem1') 
                sub_problem2.setParam('OutputFlag',0)       
                
            #申明子问题的变量
                x2 = sub_problem2.addVars(demand, vtype=grb.GRB.BINARY, name='x2')
                y2 = sub_problem2.addVars(plants, vtype=grb.GRB.BINARY, name='y2')
                
            #设置目标函数
                expr1 = grb.quicksum([(1-y2[i])*I[k,i].x for i in plants])
                expr2 = grb.quicksum([lower_demand[k,j]*(1-x2[j]) for j in demand])
                sub_problem2.setObjective(expr2 - expr1, grb.GRB.MINIMIZE)
                del expr1; del expr2 
            #载入子问题的约束
            
            #子问题约束1
                expr = grb.quicksum(x2.values())
                sub_problem2.addConstr(expr >= S)
                del expr
                
            #子问题约束2，
                for i in plants:
                    for j in demand:
                        if round(a[i,j]) == 1:
                            sub_problem2.addConstr(y2[i] >= x2[j])
                   
            #解子问题
                sub_problem2.optimize()
                
                IterSubObjVal[k,0] = sub_problem1.ObjVal
                IterSubObjVal[k,1] = sub_problem2.ObjVal
                for j in demand:
                    IterX[k,j,0] = x1[j].x
                    IterX[k,j,1] = x2[j].x
                for i in plants:
                    IterY[k,i,0] = y1[i].x
                    IterY[k,i,1] = y2[i].x
                
            #判断子问题的目标函数值是否为非负数，若是，则主问题得到了最优解,break；否则加入新的约束到主问题中
            if all(IterSubObjVal.reshape(6)>=-0.01):
                iterValue_small.append(IterSubObjVal[k,0] for k in items)
                iterValue_big.append(IterSubObjVal[k,1] for k in items)
                break
            
            else:
                for k in items:
                
                    if IterSubObjVal[k,0] < -0.001:
                        print('small_set:',IterSubObjVal[k,0])
                        iterValue_small.append(IterSubObjVal[k,0])
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = IterX[k,j,0] * a[:,j] + a_temp
                            expr2 = upper_demand[k,j]*IterX[k,j,0] + expr2
                        for i in plants:
                            if round(a_temp[i])!=0:
                                expr1 = expr1 + round(a_temp[i])/round(a_temp[i]) * I[k,i] 
                        temp_expr = expr1 >= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        constraintType.append('small')
                        del expr1;del expr2;del temp_expr;del a_temp
                        add_constraints_time += 1
                    else:
                        iterValue_small.append(np.nan)
                        
                        
                    if IterSubObjVal[k,1] < -0.001:
                        print('big_set:',IterSubObjVal[k,1])
                        iterValue_big.append(IterSubObjVal[k,1])
                        expr1 = 0; expr2 = 0; a_temp = [0]*num_plants
                        for j in demand:
                            a_temp = IterX[k,j,1] *a[:,j] + a_temp
                            expr2 = lower_demand[k,j]*(int(not IterX[k,j,1])) + expr2
                        for i in plants:
                            expr1 = expr1 + int(not a_temp[i])*I[k,i]             
                        temp_expr = expr1 <= expr2
                        master_problem.addConstr(temp_expr)    #加新的约束到主问题中
                        constraintType.append('big')
                        del expr1;del expr2;del temp_expr; del a_temp
                        add_constraints_time += 1
                    else:
                        iterValue_big.append(np.nan)
                    
    time_end = time.clock()
    
    if  master_problem.status == 2:
        
        open_DC = sum([Z[i].x for i in plants])
        I_return = [[I[k,i].x for i in plants] for k in items]
        Z_return = [Z[i].x for i in plants]
        p = [sum(I[k,i].x for i in plants) for k in items]
            
        return master_problem,constraintType,iterValue,iterValue_small,iterValue_big,[s,open_DC,I_return,Z_return,p,p/TI,master_problem.ObjVal,time_end-time_start,add_constraints_time,'optimal'] 
    
    elif master_problem.status == 3:   
        return master_problem,constraintType,iterValue,iterValue_small,iterValue_big,['-','-','-','-','-','-','-','-','-','infeasible']



    