# Psi-expander_approximation

This is the code for paper, "Prepositioning Network Design for Disaster Reliefs Stochastic Models and Psi-Expander Models Comparison", which is being reviewed at Computers & Industrial Engineering (20190904)

【Main】:DRSJTU-multi.py
    ---- It defines a class [Instance], the main part of the paper, including four kinds of different models: the deterministic model, the stochastic model, the exact \Psi-expander model proposed in [1], and the \Psi-expander model proposed in my paper.
    
【Part 1】: generator.py
    ---- It generate some input parameters for the Instance class, including graph, demand parameters, and cost parameters
    
【Part 2】: Hoeffding.py | small_big.py
    ---- It contains two models : hoeffding.py is the model in [1], which is used to compare with ours.
                                  small_big.py is the model we proposed. Don't let the name bother you, cause there are two parameters for                         this model. One is a limitation for "small constraint set" and the another for "big constraint set". hhhhh, the name is self-evident.
 
 
 What is omitted is a "Main.py" file. Because I run these code in Spyder, and no need to run in commend line or run as an exe file. Just run several seperate parts and the data would be saved in the IDE, then run the next part. 
 
 But before, please check the paper first to know what I exactly did.
 
 
 Ref:
 [1]Li Y, Shu J, Song M, et al. Multisourcing supply network design: two-stage chance-constrained model, tractable approximations, and computational results[J]. INFORMS Journal on Computing, 2017, 29(2): 287-300.
 
