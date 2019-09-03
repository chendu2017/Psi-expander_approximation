# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 19:24:38 2019

@author: chend
"""

import os 
os.chdir(r'D:\【论文】交大学报论文\代码')
import pandas as pd
import numpy as np
import small_big
import hoeffding
import simulation
import generator
from scipy import stats
from gurobipy import gurobipy as grb
INF = float('inf')

class Instance():
    def __init__(self,supplier,retailer,GRAPH,
                      demand_para,cost,radius):
        #A number
        self.supplier = supplier
        self.retailer = retailer
        
        self.radius = radius
         
        self.s = 0
        self.weight = 0
        
        #A np array
        self.lower_demand = np.asarray([demand_para[k][0].copy() for k in range(demand_para.shape[0])])
        self.mean_demand = np.asarray([demand_para[k][1].copy() for k in range(demand_para.shape[0])])
        self.upper_demand = np.asarray([demand_para[k][2].copy() for k in range(demand_para.shape[0])])
        
        self.fixed_cost = cost[0].copy()
        self.holding_cost = cost[1].copy()
        self.penalty_cost = cost[2].copy()
        self.transportation_cost = cost[3].copy()
        
        #2-dimension
        self.graph = GRAPH
    
    
    def SolveDetModel(self,demandScenario):
        
        def GetResults(self,DetModel,I,Z,Transshipment_X):
            resultsdict = {}
            plants = range(self.supplier)
            items = range(self.holding_cost.shape[0])
            resultsdict['# of Open DC'] = sum([Z[i].x for i in plants])
            resultsdict['Open DC'] = [Z[i].x for i in plants]
            resultsdict['Total Inventory'] = sum([I[k,i].x for k in items for i in plants])
            resultsdict['Inventory'] = [[I[k,i].x for i in plants] for k in items] 
            resultsdict['Fixed Cost'] = sum([Z[i].x*self.fixed_cost[i] for i in plants])
            resultsdict['Holding Cost'] = [sum([I[k,i].x*self.holding_cost[k,i] for i in plants]) for k in items]
            return resultsdict
        
        DetModel = grb.Model('DetModel') 
        DetModel.setParam('OutputFlag',0)
        DetModel.modelSense = grb.GRB.MINIMIZE
        
        plants = range(self.supplier)
        demand = range(self.retailer)
        items  = range(self.lower_demand.shape[0])
        
        I = DetModel.addVars(items,plants,vtype=grb.GRB.CONTINUOUS,name='I')
        Z = DetModel.addVars(plants,vtype=grb.GRB.BINARY,name='Z')
        Transshipment_X = DetModel.addVars(items,plants,demand,vtype=grb.GRB.CONTINUOUS, name='Transshipment_X')
        
        objFunc_holding = sum([I[k,i]*self.holding_cost[k,i] for k in items for i in plants])
        objFunc_fixed = sum([Z[i]*self.fixed_cost[i] for i in plants])
        objFunc_transportation = grb.quicksum([Transshipment_X[k,i,j]*self.transportation_cost[k,i,j] for k in items for i in plants for j in demand])
        objFunc = objFunc_holding+objFunc_fixed+objFunc_transportation
        
        DetModel.setObjective(objFunc)
        
        #添加评估模型的约束
        #约束1
        for i in plants:
            for k in items:
                DetModel.addConstr(grb.quicksum([Transshipment_X[k,i,j] for j in demand]) <= I[k,i])
        
        #约束2
        for j in demand:
            for k in items:
                DetModel.addConstr(grb.quicksum([Transshipment_X[k,i,j] for i in plants]) == demandScenario[k,j])
        
        #约束3 其实是paper中的4，gurobi自带正数约束
        for i in plants:
            for j in demand:
                if round(self.graph[i,j]) == 0:
                    for k in items:
                        DetModel.addConstr(Transshipment_X[k,i,j] <= 0)
        
        #约束4 I_i<=M*Z_i
        for i in plants:
            for k in items:
                DetModel.addConstr(I[k,i]<=100000*Z[i])
            
        
        #求解评估模型
        DetModel.optimize()
        return GetResults(self,DetModel,I,Z,Transshipment_X)
        
    
    
    def SolveStoModel(self,STOMODELCONFIGURE):
        
        def GetResults(self,I,Z,Transshipment_X,N):
            plants = range(self.supplier)
            items = range(self.lower_demand.shape[0])
            
            objVal_holding = sum([I[k,i].x*self.holding_cost[k,i] for k in items for i in plants])
            objVal_fixed = sum([Z[i].x*self.fixed_cost[i] for i in plants])
            
            resultsdict = {}
            plants = range(self.supplier)
            resultsdict['# of Open DC'] = sum([Z[i].x for i in plants])
            resultsdict['Open DC'] = [Z[i].x for i in plants]
            resultsdict['Total Inventory'] = [sum([I[k,i].x for i in plants]) for k in items]
            resultsdict['Inventory'] = [[I[k,i].x for i in plants] for k in items]
            resultsdict['Fixed Cost'] = objVal_fixed
            resultsdict['Holding Cost'] = [sum([I[k,i].x*self.holding_cost[k,i] for i in plants]) for k in items]
            return resultsdict
        
        N = STOMODELCONFIGURE['#']
        demandScenario = STOMODELCONFIGURE['demandScenario']
        
        StoModel = grb.Model('StoModel') 
        StoModel.setParam('OutputFlag',0)
        StoModel.modelSense = grb.GRB.MINIMIZE
        
        items = range(demandScenario.shape[0])
        plants = range(self.supplier)
        demand = range(self.retailer)
        
        I = StoModel.addVars(items,plants,vtype=grb.GRB.CONTINUOUS,name='I')
        Z = StoModel.addVars(plants,vtype=grb.GRB.BINARY,name='Z')
        Transshipment_X = StoModel.addVars(items,range(N),plants,demand,vtype=grb.GRB.CONTINUOUS,name='Transshipment_X')
        
        objFunc_holding = grb.quicksum([I[k,i]*self.holding_cost[k,i] for k in items for i in plants])
        objFunc_fixed = grb.quicksum([Z[i]*self.fixed_cost[i] for i in plants])
        objFunc_penalty = grb.quicksum([grb.quicksum([self.penalty_cost[k,j]*(demandScenario[k,scenario,j]-grb.quicksum([Transshipment_X[k,scenario,i,j] for i in plants])) for k in items for j in demand]) for scenario in range(N)])
        objFunc_trans = grb.quicksum([Transshipment_X[k,scenario,i,j]*self.transportation_cost[k,i,j] for scenario in range(N) for k in items for i in plants for j in demand]) #N个Scenario的transportation cost总和
        objFunc = objFunc_holding+objFunc_fixed+(objFunc_penalty+objFunc_trans)/N
        
        StoModel.setObjective(objFunc)
        '''
        TODO
        '''
        #约束1
        for i in plants:
            for k in items:
                StoModel.addConstrs((grb.quicksum([Transshipment_X[k,scenario,i,j] for j in demand]) <= I[k,i] for scenario in range(N)))
        
        #约束2
        for j in demand:
            for k in items:
                StoModel.addConstrs((grb.quicksum([Transshipment_X[k,scenario,i,j] for i in plants]) <= demandScenario[k,scenario,j] for scenario in range(N)))
        
        #约束3
        for i in plants:
            for j in demand:
                for k in items:
                    if round(self.graph[i,j]) == 0:
                        StoModel.addConstrs((Transshipment_X[k,scenario,i,j] <= 0 for scenario in range(N)))
        
        #约束4 I_i<=M*Z_i
        for i in plants:
            for k in items:
                StoModel.addConstr(I[k,i]<=100000*Z[i])
    
        #求解评估模型
        StoModel.optimize()
        return GetResults(self,I,Z,Transshipment_X,N)
    
    
    def SolveExpanderModel(self,s,S):
        master_problem,constraintType,iterValue,iterValue_small,iterValue_big,resultlist = small_big.small_big_design_multiitem(cost = [self.fixed_cost,self.holding_cost],
                                   demand = [self.lower_demand,self.mean_demand,self.upper_demand],
                                   a = self.graph,
                                   num_plants = self.supplier,
                                   num_demand = self.retailer,
                                   s = s,
                                   S = S,
                                   weight=1)
        I = resultlist[2]
        Z = resultlist[3]
        
        expanderResults = {}
        expanderResults['# of Open DC'] = resultlist[1]
        expanderResults['Open DC'] = resultlist[3]
        expanderResults['Inventory'] = resultlist[2]
        expanderResults['Total Inventory'] = resultlist[4]
        expanderResults['Fixed Cost'] = sum([Z[i]*self.fixed_cost[i] for i in range(len(Z))])
        expanderResults['Holding Cost'] = [sum(I[k][i]*self.holding_cost[k,i] for i in range(len(Z))) for k in range(3)]
        
        return master_problem,constraintType,iterValue,iterValue_small,iterValue_big,expanderResults
    
    def SolveHoeffdingModel(self,service_level):
        master_problem,constraintType,iterValue,SNum,resultlist = hoeffding.hoeffding_design_multiitem(cost = [self.fixed_cost,self.holding_cost],
                                   demand = [self.lower_demand,self.mean_demand,self.upper_demand],
                                   a = self.graph,
                                   num_plants = self.supplier,
                                   num_demand = self.retailer,
                                   service_level = service_level)
        I = resultlist[2]
        Z = resultlist[3]
        
        plants = range(self.supplier)
        items = range(self.lower_demand.shape[0])
        
        hoeffdingResults = {}
        hoeffdingResults['# of Open DC'] = resultlist[1]
        hoeffdingResults['Open DC'] = resultlist[3]
        hoeffdingResults['Inventory'] = resultlist[2]
        hoeffdingResults['Total Inventory'] = resultlist[4]
        hoeffdingResults['Fixed Cost'] = sum([Z[i]*self.fixed_cost[i] for i in plants])
        hoeffdingResults['Holding Cost'] = [sum([I[k][i]*self.holding_cost[k,i] for i in plants]) for k in items]
        
        return master_problem,constraintType,iterValue,SNum,hoeffdingResults
    
    
    
    
if __name__ == '__main__':
    items = range(3)
    
    demand_para = np.asarray([pd.read_excel(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\demand_para_Multi.xlsx',sheet_name=None)['Item{}'.format(k+1)].values for k in items])
    demandScenario = np.asarray([pd.read_excel(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\demandScenario_Multi.xlsx',sheet_name=None)['Item{}'.format(k+1)].values for k in items])
    demandScenario_forStoModel = np.asarray([pd.read_excel(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\demandScenario_forStoModel_Multi.xlsx',sheet_name=None)['Item{}'.format(k+1)].values for k in items])
    
    PENALTYCOEFLIST = [1,2,3,4,5]
    
    DETMODELRESULTS_list = []
    STOMODELRESULTS_list = []
    EXPANDERRESULTS_list = []
    HOEFFDINGRESULTS_list = []
    
    for PENALTYCOEF in PENALTYCOEFLIST:
    
        cost = []
        #fixed cost
        cost.append(np.asarray([203,193,130,117,292,174,130,157,134,161,234,220,170]))
        
        #holding cost
        F1,F2,F3 = 1,1,1
        cost.append(np.asarray([[3.40*F1,2.33*F1,2.00*F1,2.69*F1,2.63*F1,3.44*F1,3.43*F1,3.53*F1,2.33*F1,2.50*F1,3.37*F1,2.84*F1,3.76*F1],
                                [3.40*F2,2.33*F2,2.00*F2,2.69*F2,2.63*F2,3.44*F2,3.43*F2,3.53*F2,2.33*F2,2.50*F2,3.37*F2,2.84*F2,3.76*F2],
                                [3.40*F3,2.33*F3,2.00*F3,2.69*F3,2.63*F3,3.44*F3,3.43*F3,3.53*F3,2.33*F3,2.50*F3,3.37*F3,2.84*F3,3.76*F3]]))
        
        #penalty cost
        F1,F2,F3 = 3,2,1
        cost.append(np.asarray([[11.48*F1,14.32*F1,12.14*F1,16.19*F1,12.01*F1,14.90*F1,9.42*F1,11.91*F1,10.68*F1,11.24*F1,13.1*F1,11.09*F1,10.18*F1],
                                [11.48*F2,14.32*F2,12.14*F2,16.19*F2,12.01*F2,14.90*F2,9.42*F2,11.91*F2,10.68*F2,11.24*F2,13.1*F2,11.09*F2,10.18*F2],
                                [11.48*F3,14.32*F3,12.14*F3,16.19*F3,12.01*F3,14.90*F3,9.42*F3,11.91*F3,10.68*F3,11.24*F3,13.1*F3,11.09*F3,10.18*F3]])*PENALTYCOEF)
        
        #transportation_cost 
        F1,F2,F3 = 1,1,1
        cost.append(np.asarray([generator.GenerateTransportationCost()*F1,
                                generator.GenerateTransportationCost()*F2,
                                generator.GenerateTransportationCost()*F3]))
        del F1,F2,F3
        
        supplier = 13
        retailer = 13
        REPEAT = 1
        
        
        
        #---- DetModel
        for r in [1]:
            
            print('Det r={r}'.format(r=r))
            
            radius = [r]*supplier*REPEAT
            
            GRAPH = generator.GenerateRoad(radius,supplier*REPEAT,randomly=False)
            instance = Instance(supplier*REPEAT,retailer*REPEAT,GRAPH,demand_para,cost,radius)
            
            # solve model
            detResults = instance.SolveDetModel(instance.mean_demand)
            
            #simulation
            tmp = simulation.simulationWithCost_Multi(a=instance.graph,
                                          demand=demandScenario,
                                          Z=np.asarray(detResults['Open DC']),
                                          I=np.asarray(detResults['Inventory']),
                                          num_plants=supplier,num_demand=retailer,
                                          tranposrtation_cost=instance.transportation_cost,
                                          penalty_cost=instance.penalty_cost,
                                          cishu=demandScenario.shape[1])
            
            detResults['Type1 Service Level'] = tmp[0]
            detResults['Type2 Service Level'] = tmp[1]
            detResults['Type1 Service Level-worst'] = tmp[2]
            detResults['Type2 Service Level-worst'] = tmp[3]
            detResults['Transportation Cost'] = tmp[4]
            detResults['Penalty Cost'] = tmp[5]
            detResults['r'] = r
            detResults['PENALTYCOEF'] = PENALTYCOEF
    
            DETMODELRESULTS_list.append(detResults)
                
        print('~~~~DetModel Finished~~~~')    
        
        
        #---- StoModel
        for r in [1]:
                
            print('Sto r={r}'.format(r=r))
            
            radius = [r]*supplier*REPEAT
            
            GRAPH = generator.GenerateRoad(radius,supplier*REPEAT,randomly=False)
            instance = Instance(supplier*REPEAT,retailer*REPEAT,GRAPH,demand_para,cost,radius)
            
            # solve model
            STOMODELCONFIGURE = {
                                '#':demandScenario_forStoModel.shape[1], # number of scenario  
                                'demandScenario':demandScenario_forStoModel
                                }
            stoResults = instance.SolveStoModel(STOMODELCONFIGURE)
            #simulation
            tmp = simulation.simulationWithCost_Multi(a=instance.graph,
                                          demand=demandScenario,
                                          Z=np.asarray(stoResults['Open DC']),
                                          I=np.asarray(stoResults['Inventory']),
                                          num_plants=supplier,num_demand=retailer,
                                          tranposrtation_cost=instance.transportation_cost,
                                          penalty_cost=instance.penalty_cost,
                                          cishu=demandScenario.shape[1])
            stoResults['Type1 Service Level'] = tmp[0]
            stoResults['Type2 Service Level'] = tmp[1]
            stoResults['Type1 Service Level-worst'] = tmp[2]
            stoResults['Type2 Service Level-worst'] = tmp[3]
            stoResults['Transportation Cost'] = tmp[4]
            stoResults['Penalty Cost'] = tmp[5]
            stoResults['r'] = r
            stoResults['PENALTYCOEF'] = PENALTYCOEF
            STOMODELRESULTS_list.append(stoResults)
                
        print('~~~~StoModel Finished~~~~')
        
        #---- Psi-expander Model
        for s in range(10,11): #range(11,13)
            for S in range(5,6): #range(2,4)
                for r in [1]:
    
                    print('Exp r={r},s={s}, S={S}'.format(r=r,s=s,S=S))
                
                    radius = [r]*supplier*REPEAT
                    
                    GRAPH = generator.GenerateRoad(radius,supplier*REPEAT,randomly=False)
                    instance = Instance(supplier*REPEAT,retailer*REPEAT,GRAPH,demand_para,cost,radius)
                    
                     #防止无解,程序停止
                    try:
                        expanderResults = instance.SolveExpanderModel(s,S)[-1]
                        tmp = simulation.simulationWithCost_Multi(a=instance.graph,
                                      demand=demandScenario,
                                      Z=np.asarray(expanderResults['Open DC']),
                                      I=np.asarray(expanderResults['Inventory']),
                                      num_plants=supplier,num_demand=retailer,
                                      tranposrtation_cost=instance.transportation_cost,
                                      penalty_cost=instance.penalty_cost,
                                      cishu=demandScenario.shape[1])
                        expanderResults['Type1 Service Level'] = tmp[0]
                        expanderResults['Type2 Service Level'] = tmp[1]
                        expanderResults['Type1 Service Level-worst'] = tmp[2]
                        expanderResults['Type2 Service Level-worst'] = tmp[3]
                        expanderResults['Transportation Cost'] = tmp[4]
                        expanderResults['Penalty Cost'] = tmp[5]
                        expanderResults['r'] = r
                        expanderResults['s'] = s
                        expanderResults['S'] = S
                        expanderResults['PENALTYCOEF'] = PENALTYCOEF
                        EXPANDERRESULTS_list.append(expanderResults)
                    except:
                        pass
        
        print('~~~~ExpanderModel Finished~~~~')
                    
        #---- HoeffdingModel  
        for r in [1]:
                
            print('Hoef r={r}'.format(r=r))
            
            radius = [r]*supplier*REPEAT
            
            GRAPH = generator.GenerateRoad(radius,supplier*REPEAT,randomly=False)
            instance = Instance(supplier*REPEAT,retailer*REPEAT,GRAPH,demand_para,cost,radius)
        
            #防止无解,程序停止
            try:
                hoeffdingResults = instance.SolveHoeffdingModel(service_level=0.1)[-1] #-1是结果的字典
                tmp = simulation.simulationWithCost_Multi(a=instance.graph,
                              demand=demandScenario,
                              Z=np.asarray(hoeffdingResults['Open DC']),
                              I=np.asarray(hoeffdingResults['Inventory']),
                              num_plants=supplier,num_demand=retailer,
                              tranposrtation_cost=instance.transportation_cost,
                              penalty_cost=instance.penalty_cost,
                              cishu=demandScenario.shape[1])
                hoeffdingResults['Type1 Service Level'] = tmp[0]
                hoeffdingResults['Type2 Service Level'] = tmp[1]
                hoeffdingResults['Type1 Service Level-worst'] = tmp[2]
                hoeffdingResults['Type2 Service Level-worst'] = tmp[3]
                hoeffdingResults['Transportation Cost'] = tmp[4]
                hoeffdingResults['Penalty Cost'] = tmp[5]
                hoeffdingResults['r'] = r
                hoeffdingResults['PENALTYCOEF'] = PENALTYCOEF
                
            except:
                pass
            HOEFFDINGRESULTS_list.append(hoeffdingResults)
        
        print('~~~~HoeffdingModel Finished~~~~')
        
    #存储Det结果
        DETMODELRESULTS_DF = pd.DataFrame(DETMODELRESULTS_list)
    #存储Sto结果
        STOMODELRESULTS_DF = pd.DataFrame(STOMODELRESULTS_list)
    #存储ExpM结果
        EXPANDERRESULTS_DF = pd.DataFrame([item for item in EXPANDERRESULTS_list if len(item)>0])
    #存储Hoeffding结果
        HOEFFDINGRESULTS_DF = pd.DataFrame(HOEFFDINGRESULTS_list)           
        
    DETMODELRESULTS_DF.to_excel(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\multi\Det_Multi.xlsx',index=False)            
    STOMODELRESULTS_DF.to_excel(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\multi\Sto_Multi.xlsx',index=False)
    EXPANDERRESULTS_DF.to_excel(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\multi\Expander_Multi.xlsx',index=False)
    HOEFFDINGRESULTS_DF.to_excel(r'D:\【论文】交大学报论文\二轮s20190707\数值结果\multi\Hoeffding_Multi.xlsx',index=False)            
                  
            
            
            
    