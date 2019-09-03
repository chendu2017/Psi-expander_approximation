# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 18:04:19 2018

@author: chend
"""
import numpy as np
import math
from scipy import stats
import pandas as pd

def GenerateRoad(r,NumNodes,randomly=True):
    if randomly:
        graph = np.random.random(size=(NumNodes,NumNodes))+0.5
        plants = range(NumNodes)
        demand = range(NumNodes)
        a = np.array([[1]*NumNodes]*NumNodes)
        for i in range(NumNodes):
            graph[i,i] = 0
        
    else:
        INF = float("inf")
        
        #Matriz de distancias entre cada Nó. INF = Nós não possuem ligação.
        graph =  np.asarray(
                [  [0,75.1,31.3,70.5,92.4,47.7,120,98.9,173,142,188,234,220],
                    [0,0,65.3,145,26.7,116,188,133,107,210,188,308,152],
                    [0,0,0,101,82.7,71.8,145,69.7,163,166,219,265,210],
                    [0,0,0,0,163,117,190,169,243,212,118,164,290],
                    [0,0,0,0,0,133,206,150,80.5,227,280,327,186],
                    
                    [0,0,0,0,0,0,72,140,214,95.1,236,295,261],
                    [0,0,0,0,0,0,0,212,286,170,308,281,333],
                    [0,0,0,0,0,0,0,0,231,234,287,333,278],
                    [0,0,0,0,0,0,0,0,0,308,361,407,242],
                    [0,0,0,0,0,0,0,0,0,0,330,390,355],
                    
                    [0,0,0,0,0,0,0,0,0,0,0,49.6,408],
                    [0,0,0,0,0,0,0,0,0,0,0,0,454],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0],
                ])
    
        for i in range(13):
            for j in range(i,13):
                graph[j][i] = graph[i][j]
    
        graph = graph/(np.sum(graph)/(13*12))
    
        plants = range(13)
        demand = range(13)
    
        a = np.array([[1]*13]*13)
    #计算每个节点的半径
    for i in plants:
        for j in demand:
            a[i,j] = a[i,j]*(graph[(i,j)] <= r[i] )
            
    return a

def GenerateScenario(N,demand_para):
    retailer = 13
    demandScenario = np.zeros(shape=(N,retailer))    
    for k in range(retailer):
        lower, upper = demand_para[0,k], demand_para[2,k]
        mu, sigma = demand_para[1,k], 10
        demandScenario[:,k] = np.round(stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N))
    
    return demandScenario

def GenerateScenarioWithMixedDist(demand,num_demand,cishu):
    lower_demand = demand[0]
    mean_demand = demand[1]
    upper_demand = demand[2]
    
    realized_demand = np.zeros(shape=(cishu,num_demand))
    
    for k in range(num_demand):
        lower, upper = lower_demand[k], upper_demand[k]
        
        #将Scenario分为4个部分，分别服从 normal, uniform, triangular, two-point
        
        # - normal
        # 由于事前的方差设定为20，但有可能不对，因此将此块分为10,15,20,25,30的方差，分别generate scenario
        for coef in [2,3,4,5,6]:
            mu, sigma = mean_demand[k], 5*coef
            realized_demand[int((coef-2)*cishu/4/5):int((coef-1)*cishu/4/5),k] = np.round(stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=int(cishu/4/5)),2)
        
        # - uniform
        realized_demand[int(cishu/4):int(2*cishu/4),k] = np.random.randint(lower_demand[k],upper_demand[k]+1,size=int(cishu/4))
        
        # - triangular 
        realized_demand[int(2*cishu/4):int(3*cishu/4),k] = np.round(np.random.triangular(lower_demand[k],mean_demand[k],upper_demand[k],int(cishu/4)))
        
        # - two-point 
        realized_demand[int(3*cishu/4):,k] = lower_demand[k] + np.random.randint(0,2,int(cishu/4))*(upper_demand[k]-lower_demand[k])
            
    return np.round(realized_demand)

def GenerateTransportationCost():
    #Matriz de distancias entre cada Nó. INF = Nós não possuem ligação.
    graph =  np.asarray([
            [0.  , 0.89, 0.35, 0.8 , 1.4 , 0.52, 1.32, 1.11, 2.3 , 1.58, 2.11, 4.26, 2.6 ],
            [0.89, 0.  , 0.78, 1.69, 0.51, 1.41, 2.21, 1.54, 1.41, 2.47, 2.27, 5.15, 1.71],
            [0.35, 0.78, 0.  , 1.15, 1.29, 0.87, 1.67, 0.76, 2.19, 1.93, 2.46, 4.61, 2.49],
            [0.8 , 1.69, 1.15, 0.  , 2.2 , 1.32, 2.12, 1.91, 3.1 , 2.38, 1.31, 5.06, 1.87],
            [1.4 , 0.51, 1.29, 2.2 , 0.  , 1.92, 2.72, 1.24, 0.9 , 2.98, 2.78, 5.66, 2.22],
            [0.52, 1.41, 0.87, 1.32, 1.92, 0.  , 0.8 , 1.63, 2.82, 1.06, 2.63, 3.74, 3.12],
            [1.32, 2.21, 1.67, 2.12, 2.72, 0.8 , 0.  , 2.43, 3.62, 1.86, 3.43, 2.94, 3.92],
            [1.11, 1.54, 0.76, 1.91, 1.24, 1.63, 2.43, 0.  , 2.14, 2.69, 3.22, 5.37, 3.25],
            [2.3 , 1.41, 2.19, 3.1 , 0.9 , 2.82, 3.62, 2.14, 0.  , 3.88, 3.68, 6.56, 3.12],
            [1.58, 2.47, 1.93, 2.38, 2.98, 1.06, 1.86, 2.69, 3.88, 0.  , 3.69, 4.8 , 4.18],
            [2.11, 2.27, 2.46, 1.31, 2.78, 2.63, 3.43, 3.22, 3.68, 3.69, 0.  , 6.37, 0.56],
            [4.26, 5.15, 4.61, 5.06, 5.66, 3.74, 2.94, 5.37, 6.56, 4.8 , 6.37, 0.  , 6.86],
            [2.6 , 1.71, 2.49, 1.87, 2.22, 3.12, 3.92, 3.25, 3.12, 4.18, 0.56, 6.86, 0.  ]
            ])


    return graph


def GenerateDemandPara_DemandScenario_DemandForSto():
    SCENARIO_forSimulation = 3000
    SCENARIO_forStoModel = 200
    
    supplier = 13
    retailer = 13
    REPEAT = 1

    demand_para= np.array([[50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50,  50],
                          [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                          [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150]])

    demandScenario = GenerateScenarioWithMixedDist(demand_para,retailer,SCENARIO_forSimulation)
    
    demandScenario_forStoModel = GenerateScenario(SCENARIO_forStoModel,demand_para)
    
        
        
    with pd.ExcelWriter(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\demandScenario.xlsx') as writer:
          pd.DataFrame(demandScenario,
                         columns=['Node{}'.format(j+1) for j in range(retailer)]).to_excel(writer, sheet_name='Sheet1',index=False) 
                
    with pd.ExcelWriter(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\demandScenario_forStoModel.xlsx') as writer:
        pd.DataFrame(demandScenario_forStoModel,
                         columns=['Node{}'.format(j+1) for j in range(retailer)]).to_excel(writer, sheet_name='Sheet1',index=False) 
        
    with pd.ExcelWriter(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\demand_para.xlsx') as writer:
        pd.DataFrame(demand_para,
                     columns=['Node{}'.format(j+1) for j in range(retailer)]).to_excel(writer, sheet_name='Sheet1',index=False) 
             
        
        
def GenerateDemandPara_DemandScenario_DemandForSto_Multi():
    SCENARIO_forSimulation = 3000
    SCENARIO_forStoModel = 200
    
    supplier = 13
    retailer = 13


    demand_para= np.asarray([np.array([[50+k,  50+k,  50+k,  50+k,  50+k,  50+k,  50+k,  50+k,  50+k,  50+k,  50+k,  50+k,  50+k],
                          [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                          [150-k, 150-k, 150-k, 150-k, 150-k, 150-k, 150-k, 150-k, 150-k, 150-k, 150-k, 150-k, 150-k]])
                for k in [-10,0,10]])

    demandScenario = np.asarray([GenerateScenarioWithMixedDist(demand_para[k],retailer,SCENARIO_forSimulation)
                                 for k in [0,1,2]])
    
    demandScenario_forStoModel = np.asarray([GenerateScenario(SCENARIO_forStoModel,demand_para[k]) 
                                                for k in [0,1,2]])
    
        
        
    with pd.ExcelWriter(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\demandScenario_Multi.xlsx') as writer:
        for k in [0,1,2]:
            pd.DataFrame(demandScenario[k],
                             columns=['Node{}'.format(j+1) for j in range(retailer)]).to_excel(writer, sheet_name='Item{}'.format(k+1),index=False)
                
    with pd.ExcelWriter(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\demandScenario_forStoModel_Multi.xlsx') as writer:
        for k in [0,1,2]:
            pd.DataFrame(demandScenario_forStoModel[k],
                         columns=['Node{}'.format(j+1) for j in range(retailer)]).to_excel(writer, sheet_name='Item{}'.format(k+1),index=False) 
        
    with pd.ExcelWriter(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\demand_para_Multi.xlsx') as writer:
        for k in [0,1,2]:
            pd.DataFrame(demand_para[k],
                     columns=['Node{}'.format(j+1) for j in range(retailer)]).to_excel(writer, sheet_name='Item{}'.format(k+1),index=False) 
             
            
            

