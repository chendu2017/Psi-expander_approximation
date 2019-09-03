# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:43:26 2018

@author: chend
"""

import random
from gurobipy import gurobipy as grb
from scipy import stats
import numpy as np
import pandas as pd
def uniform_simulation(a,demand,Z,I,num_plants,num_demand,cishu=500):
    print('--------uniform_simulation----------')
    lower_demand = demand[0]
    upper_demand = demand[2]
    
    plants = range(num_plants)
    demand = range(num_demand)
    
    results = []

    for n in range(cishu):
        realized_demand = [0] *len(lower_demand)
        for k in range(len(realized_demand)):
            realized_demand[k] = round(random.randint(lower_demand[k],upper_demand[k]))
            
        #申明Fillrate中Z(w)值的那个模型    
        evaluation_problem = grb.Model('evaluation_problem') 
        evaluation_problem.setParam('OutputFlag',0)
        evaluation_problem.modelSense = grb.GRB.MAXIMIZE
        
        evaluation_X = evaluation_problem.addVars(plants, demand, obj = [[1 for j in demand] for i in plants], vtype=grb.GRB.INTEGER, name='evaluation_X')
        
        #添加评估模型的约束
        #约束1
        for i in plants:
            expr = 0
            for j in demand:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= I[i])
            del expr
        
        #约束2
        for j in demand:
            expr = 0
            for i in plants:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= realized_demand[j])
            del expr
        
        #约束3 其实是paper中的4，gurobi自带正数约束
        for i in plants:
            for j in demand:
                if round(a[i,j]*Z[i]) == 0:
                    evaluation_problem.addConstr(evaluation_X[i,j] <= 0)
        
        #求解评估模型，并求出fillrate 还要输出
        evaluation_problem.optimize()
        
        type1ServiceLevel = sum([np.abs(sum([evaluation_X[i,j].x for i in plants])-realized_demand[j])<=0.005 for j in demand])/num_demand
        type2ServiceLevel = sum([sum([evaluation_X[i,j].x for i in plants]) for j in demand])/sum(realized_demand)

        results.append([type1ServiceLevel,type2ServiceLevel])
    return np.mean(np.asarray(results),axis=0)


def triangular_simulation(a,demand,Z,I,mode,num_plants,num_demand,cishu=500):
    print('--------triangular_simulation----------')
    lower_demand = demand[0]
    mean_demand = demand[1]
    upper_demand = demand[2]
    
    total_CC = 0
    total_fillrate = 0 
    total_arcs = 0
    plants = range(num_plants)
    demand = range(num_demand)
    
    for n in range(cishu):
        
        realized_demand = [0] *len(lower_demand)
        for k in range(len(realized_demand)):
            realized_demand[k] = round(random.triangular(lower_demand[k],upper_demand[k],mean_demand[k]*mode))
            
        #申明Fillrate中Z(w)值的那个模型    
        evaluation_problem = grb.Model('evaluation_problem') 
        evaluation_problem.setParam('OutputFlag',0)
        evaluation_problem.modelSense = grb.GRB.MAXIMIZE
        
        #申明评估模型的变量
        evaluation_X = evaluation_problem.addVars(plants, demand, obj = [[1 for j in demand] for i in plants], vtype=grb.GRB.INTEGER, name='evaluation_X')
        
        #添加评估模型的约束
        #约束1
        for i in plants:
            expr = 0
            for j in demand:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= I[i])
            del expr
        
        #约束2
        for j in demand:
            expr = 0
            for i in plants:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= realized_demand[j])
            del expr
        
        #约束3 其实是paper中的4，gurobi自带正数约束
        for i in plants:
            for j in demand:
                if round(a[i,j]*Z[i]) == 0:
                    evaluation_problem.addConstr(evaluation_X[i,j] <= 0)
        
        #求解评估模型，并求出fillrate 还要输出
        evaluation_problem.optimize()
        
        #########################运输完成了
        #########################计算用上的道路arcs
        arcs = 0
        for i in plants:
            for j in demand:
                arcs = (round(evaluation_X[i,j].x) > 0.1) + arcs
                
        ########################CC的值##########################
        ##################来看是否满足cutting plane的LP
        #申明CC问题    
        CC_problem = grb.Model('CC_problem') 
        CC_problem.setParam('OutputFlag',0)       
        
        #申明CC问题的变量
        CC_x = CC_problem.addVars(demand, vtype=grb.GRB.BINARY, name='CC_x')
        CC_y = CC_problem.addVars(plants, vtype=grb.GRB.BINARY, name='CC_y')
        
        #设置目标函数
        expr1 = 0; expr2 = 0
        for i in plants:
            expr1 = expr1 + CC_y[i] * I[i]
        for j in demand:
            expr2 = expr2 + realized_demand[j] * CC_x[j]
        CC_problem.setObjective(expr1 - expr2 , grb.GRB.MINIMIZE)
        del expr1; del expr2 
        #载入CC问题的约束
        
        #约束1
        for i in plants:
            for j in demand:
                if round(a[i,j]*round(Z[i])) == 1:
                    CC_problem.addConstr(CC_y[i] >= CC_x[j])
                
        #解CC问题
        CC_problem.optimize()
    
        total_CC += (CC_problem.ObjVal>=0)
        total_fillrate += evaluation_problem.ObjVal/sum(realized_demand)
        total_arcs +=arcs
        if n%100 == 0:
            print(n)
    
    eva_X = np.zeros(shape=(num_plants,num_demand))
    for i in plants:
        for i in demand:
            eva_X[i,j] = round(evaluation_X[i,j].x)
    return eva_X,total_arcs/cishu,total_CC/cishu,total_fillrate/cishu
    

def normal_simulation(a,demand,Z,I,num_plants,num_demand,tranposrtation_cost,penalty_cost,cishu=500):
    print('--------Mixed_distribution_simulation----------')
    #lower_demand = demand[0]
    #mean_demand = demand[1]
    #upper_demand = demand[2]
    
    realized_demand = np.zeros(shape=(cishu,num_demand))
    realized_demand = demand
    
    plants = range(num_plants)
    demand = range(num_demand)
    
    results = []
        
    for n in range(cishu):  
        oneDemandScenario = realized_demand[n]

        #申明Fillrate中Z(w)值的那个模型    
        evaluation_problem = grb.Model('evaluation_problem') 
        evaluation_problem.setParam('OutputFlag',0)
        evaluation_problem.modelSense = grb.GRB.MAXIMIZE
        
        evaluation_X = evaluation_problem.addVars(plants, demand, obj = [[1 for j in demand] for i in plants], vtype=grb.GRB.CONTINUOUS, name='evaluation_X')
        
        #添加评估模型的约束
        #约束1
        for i in plants:
            expr = 0
            for j in demand:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= I[i])
            del expr
        
        #约束2
        for j in demand:
            expr = 0
            for i in plants:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= oneDemandScenario[j])
            del expr
        
        #约束3 其实是paper中的4，gurobi自带正数约束
        for i in plants:
            for j in demand:
                if round(a[i,j]*Z[i]) == 0:
                    evaluation_problem.addConstr(evaluation_X[i,j] <= 0)
        
        #求解评估模型，并求出fillrate 还要输出
        evaluation_problem.optimize()
        
        #transportation 
        trans = np.zeros(shape=(num_plants,num_demand))
        for i in plants:
            for j in demand:
                trans[i,j] = evaluation_X[i,j].x
                
        type1ServiceLevel = sum(np.abs(trans.sum(axis=0)-oneDemandScenario)<=0.0005)/num_demand
        type2ServiceLevel = trans.sum()/sum(oneDemandScenario)
        trans_cost = (trans*tranposrtation_cost).sum()
        penal_cost = sum((oneDemandScenario-trans.sum(axis=0))*penalty_cost)
        results.append([type1ServiceLevel,type2ServiceLevel,trans_cost,penal_cost])

    type1ServiceLevel_mean,type2ServiceLevel_mean,trans_cost_mean,penal_cost_mean = np.mean(np.asarray(results),axis=0)
    type1ServiceLevel_worst,type2ServiceLevel_worst,trans_cost_best,penal_cost_best = np.amin(np.asarray(results),axis=0)
    return type1ServiceLevel_mean,type2ServiceLevel_mean,type1ServiceLevel_worst,type2ServiceLevel_worst,trans_cost_mean,penal_cost_mean

def two_point_simulation(a,demand,Z,I,num_plants,num_demand,cishu=500):
    print('--------two_point_simulation----------')
    lower_demand = demand[0]
    upper_demand = demand[2]
    
    total_CC = 0
    total_fillrate = 0 
    total_arcs = 0
    plants = range(num_plants)
    demand = range(num_demand)
    
    for n in range(cishu):
        
        realized_demand = [0] *len(lower_demand)
        for k in range(len(realized_demand)):
            realized_demand[k] = round(lower_demand[k] + random.randint(0,1)*(upper_demand[k]-lower_demand[k]))
            
        #申明Fillrate中Z(w)值的那个模型    
        evaluation_problem = grb.Model('evaluation_problem') 
        evaluation_problem.setParam('OutputFlag',0)
        evaluation_problem.modelSense = grb.GRB.MAXIMIZE
        
        evaluation_X = evaluation_problem.addVars(plants, demand, obj = [[1 for j in demand] for i in plants], vtype=grb.GRB.INTEGER, name='evaluation_X')
        
        #添加评估模型的约束
        #约束1
        for i in plants:
            expr = 0
            for j in demand:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= I[i])
            del expr
        
        #约束2
        for j in demand:
            expr = 0
            for i in plants:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= realized_demand[j])
            del expr
        
        #约束3 其实是paper中的4，gurobi自带正数约束
        for i in plants:
            for j in demand:
                if round(a[i,j]*Z[i]) == 0:
                    evaluation_problem.addConstr(evaluation_X[i,j] <= 0)
        
        #求解评估模型，并求出fillrate 还要输出
        evaluation_problem.optimize()
        
        #########################运输完成了
        #########################计算用上的道路arcs
        arcs = 0
        for i in plants:
            for j in demand:
                arcs = (round(evaluation_X[i,j].x) > 0.1) + arcs
       
        ########################CC的值##########################
        ##################来看是否满足cutting plane的LP
        #申明CC问题    
        CC_problem = grb.Model('CC_problem') 
        CC_problem.setParam('OutputFlag',0)       
        
        #申明CC问题的变量
        CC_x = CC_problem.addVars(demand, vtype=grb.GRB.BINARY, name='CC_x')
        CC_y = CC_problem.addVars(plants, vtype=grb.GRB.BINARY, name='CC_y')
        
        #设置目标函数
        expr1 = 0; expr2 = 0
        for i in plants:
            expr1 = expr1 + CC_y[i] * I[i]
        for j in demand:
            expr2 = expr2 + realized_demand[j] * CC_x[j]
        CC_problem.setObjective(expr1 - expr2 , grb.GRB.MINIMIZE)
        del expr1; del expr2 
        #载入CC问题的约束
        
        #约束1
        for i in plants:
            for j in demand:
                if round(a[i,j]*round(Z[i])) == 1:
                    CC_problem.addConstr(CC_y[i] >= CC_x[j])
                
        #解CC问题
        CC_problem.optimize()
    
        total_CC += (CC_problem.ObjVal>=0)
        total_fillrate += evaluation_problem.ObjVal/sum(realized_demand)
        total_arcs +=arcs
        if n%100 == 0:
            print(n)
    
    eva_X = np.zeros(shape=(num_plants,num_demand))
    for i in plants:
        for j in demand:
            eva_X[i,j] = round(evaluation_X[i,j].x)
    return eva_X,total_arcs/cishu,total_CC/cishu,total_fillrate/cishu



def simulationWithCost(a,demand,Z,I,num_plants,num_demand,tranposrtation_cost,penalty_cost,cishu=500):
    print('--------Mixed_distribution_simulation----------')
    #lower_demand = demand[0]
    #mean_demand = demand[1]
    #upper_demand = demand[2]
    
    realized_demand = np.zeros(shape=(cishu,num_demand))
    realized_demand = demand
    
    plants = range(num_plants)
    demand = range(num_demand)
    
    results = []
        
    for n in range(cishu):  
        oneDemandScenario = realized_demand[n]

        #申明Fillrate中Z(w)值的那个模型    
        evaluation_problem = grb.Model('evaluation_problem') 
        evaluation_problem.setParam('OutputFlag',0)
        evaluation_problem.modelSense = grb.GRB.MINIMIZE
        
        evaluation_X = evaluation_problem.addVars(plants, demand, vtype=grb.GRB.CONTINUOUS, name='evaluation_X')
        
        objFunc_penal = grb.quicksum([penalty_cost[j]*(oneDemandScenario[j] - grb.quicksum([evaluation_X[i,j] for i in plants])) for j in demand])
        objFunc_trans = grb.quicksum(evaluation_X[i,j]*tranposrtation_cost[i,j] for i in plants for j in demand)
        evaluation_problem.setObjective(objFunc_penal + objFunc_trans)
        
        #添加评估模型的约束
        #约束1
        for i in plants:
            expr = 0
            for j in demand:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= I[i])
            del expr
        
        #约束2
        for j in demand:
            expr = 0
            for i in plants:
                expr = expr + evaluation_X[i,j]
            evaluation_problem.addConstr(expr <= oneDemandScenario[j])
            del expr
        
        #约束3 其实是paper中的4，gurobi自带正数约束
        for i in plants:
            for j in demand:
                if round(a[i,j]*Z[i]) == 0:
                    evaluation_problem.addConstr(evaluation_X[i,j] <= 0)
        
        #求解评估模型，并求出fillrate 还要输出
        evaluation_problem.optimize()
        
        #transportation 
        trans = np.zeros(shape=(num_plants,num_demand))
        for i in plants:
            for j in demand:
                trans[i,j] = evaluation_X[i,j].x
                
        type1ServiceLevel = sum(np.abs(trans.sum(axis=0)-oneDemandScenario)<=0.0005)/num_demand
        type2ServiceLevel = trans.sum()/sum(oneDemandScenario)
        trans_cost = (trans*tranposrtation_cost).sum()
        penal_cost = sum((oneDemandScenario-trans.sum(axis=0))*penalty_cost)
        results.append([type1ServiceLevel,type2ServiceLevel,trans_cost,penal_cost])

    type1ServiceLevel_mean,type2ServiceLevel_mean,trans_cost_mean,penal_cost_mean = np.mean(np.asarray(results),axis=0)
    type1ServiceLevel_worst,type2ServiceLevel_worst,trans_cost_best,penal_cost_best = np.amin(np.asarray(results),axis=0)
    return type1ServiceLevel_mean,type2ServiceLevel_mean,type1ServiceLevel_worst,type2ServiceLevel_worst,trans_cost_mean,penal_cost_mean

    
def simulationWithCost_Multi(a,demand,Z,I,num_plants,num_demand,tranposrtation_cost,penalty_cost,cishu=500):
    print('--------Mixed_distribution_simulation----------')
    #lower_demand = demand[0]
    #mean_demand = demand[1]
    #upper_demand = demand[2]
    
    realized_demand = np.zeros(shape=(cishu,num_demand))
    realized_demand = demand
    
    items  = range(realized_demand.shape[0])
    plants = range(num_plants)
    demand = range(num_demand)
    
    results = []
    trans_cost_array = np.zeros(shape=(cishu,realized_demand.shape[0]))
    penal_cost_array = np.zeros(shape=(cishu,realized_demand.shape[0]))
        
    for n in range(cishu):  
        oneDemandScenario = realized_demand[:,n,:]

        #申明Fillrate中Z(w)值的那个模型    
        evaluation_problem = grb.Model('evaluation_problem') 
        evaluation_problem.setParam('OutputFlag',0)
        evaluation_problem.modelSense = grb.GRB.MINIMIZE
        
        evaluation_X = evaluation_problem.addVars(items, plants, demand, vtype=grb.GRB.CONTINUOUS, name='evaluation_X')
        
        objFunc_penal = grb.quicksum([penalty_cost[k,j]*(oneDemandScenario[k,j] - grb.quicksum([evaluation_X[k,i,j] for i in plants])) for k in items for j in demand])
        objFunc_trans = grb.quicksum(evaluation_X[k,i,j]*tranposrtation_cost[k,i,j] for k in items for i in plants for j in demand)
        evaluation_problem.setObjective(objFunc_penal + objFunc_trans)
        
        #添加评估模型的约束
        #约束1
        for k in items:
            for i in plants:
                evaluation_problem.addConstr( grb.quicksum([evaluation_X[k,i,j] for j in demand]) <= I[k,i])
        
        #约束2
        for k in items:
            for j in demand:
                evaluation_problem.addConstr( grb.quicksum([evaluation_X[k,i,j] for i in plants]) <= oneDemandScenario[k,j])

        
        #约束3 其实是paper中的4，gurobi自带正数约束
        for i in plants:
            for j in demand:
                if round(a[i,j]*Z[i]) == 0:
                    for k in items:
                        evaluation_problem.addConstr(evaluation_X[k,i,j] <= 0)
        
        #求解评估模型，并求出fillrate 还要输出
        evaluation_problem.optimize()
        
        #transportation 
        trans = np.zeros(shape=(realized_demand.shape[0],num_plants,num_demand))
        for k in items:
            for i in plants:
                for j in demand:
                    trans[k,i,j] = evaluation_X[k,i,j].x
                
        type1ServiceLevel = (np.abs(trans.sum(axis=1)-oneDemandScenario)<=0.0005).sum()/(num_demand*realized_demand.shape[0])
        type2ServiceLevel = trans.sum()/oneDemandScenario.sum()
        trans_cost = (trans*tranposrtation_cost).sum(axis=1).sum(axis=1)
        penal_cost = ((oneDemandScenario-trans.sum(axis=1))*penalty_cost).sum(axis=1)
        results.append([type1ServiceLevel,type2ServiceLevel,trans_cost.sum(),penal_cost.sum()])
        trans_cost_array[n,:] = trans_cost
        penal_cost_array[n,:] = penal_cost
        
    type1ServiceLevel_mean,type2ServiceLevel_mean,trans_cost_mean,penal_cost_mean = np.mean(np.asarray(results),axis=0)
    trans_cost_mean = [cost for cost in np.mean(trans_cost_array,axis=0)]
    penal_cost_mean = [cost for cost in np.mean(penal_cost_array,axis=0)]
    type1ServiceLevel_worst,type2ServiceLevel_worst,trans_cost_best,penal_cost_best = np.amin(np.asarray(results),axis=0)
    return type1ServiceLevel_mean,type2ServiceLevel_mean,type1ServiceLevel_worst,type2ServiceLevel_worst,trans_cost_mean,penal_cost_mean
