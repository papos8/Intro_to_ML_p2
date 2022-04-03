import numpy as np
import pandas as pd
from data_preprocessing import *
import matplotlib.pyplot as plt
from scipy.linalg import svd

def data_import():
    X = np.empty((1014, 7))
    for i, col_id in enumerate(range(1,8)):
        X[:, i] = np.asarray(raw_data[:,col_id-1])
    return X

if __name__ == "__main__": #Stops python from running it when importing in a new script
    """
    Once we have load the data we visualize and apply PCA


    """
    plt.figure(figsize=(40,40))
    k=1
    for i in range(7):    
        for j in range(7):
            plt.subplot(7,7,k)
            for c in range(1,len(classes)+1):
                    # select indices belonging to class c:
                    class_mask = y==c                   
                    plt.plot(data[class_mask,i], data[class_mask,j], 'o',alpha=1)
                    plt.legend(classNames)
                    plt.xlabel(attributeNames[i])
                    plt.ylabel(attributeNames[j])
            k=k+1   
               
    #Create X matrix from raw_data
    
    X = data_import()


    #Substract the mean / SVD
    Y = X - np.ones((N,1))*X.mean(axis=0)     
    U,S,V = svd(Y,full_matrices=False)          
    rho = (S*S) / (S*S).sum() 
    threshold = 0.9

    # Compute values of N, M and C.
    N = len(y)
    M = len(attributeNames)
    C = len(classNames)

    #plot variance explained
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.show()

    # PCA by computing SVD of Y
    U,S,Vh = svd(Y,full_matrices=False)

    V = Vh.T   

    # Project the centered data onto principal component space
    Z = Y @ V

    # Indices of the principal components to be plotted
    i = 0
    j = 1

    # Plot PCA of the data
    f = plt.figure()
    plt.title('Maternal Health Risks: PCA')
    #Z = array(Z)
    for c in range(3):
        # select indices belonging to class c:
        class_mask = y==c+1
        plt.plot(Z[class_mask,i], Z[class_mask,j], 'o')
        print(Z[class_mask,i])
    plt.legend(classNames)
    plt.xlabel('PC{0}'.format(i+1))
    plt.ylabel('PC{0}'.format(j+1))

    # Output result to screen
    plt.show()

    #Histogram for each attribute
    for i in range(7):
        f = plt.figure()
        plt.title('Normal distribution')
        plt.xlabel(attributeNames[i])
        plt.hist(X[:,i], bins=20, density=True)

    #Boxplot - Attributes

    ##Tried to plot them all together
    #mask1=[0,1,2,5]

    #fig1 = plt.boxplot(X[:,mask1])
    #plt.xticks(range(1,5),["Age","Systolic BP", "Diastolic BP", "Heart Rate"],rotation=45,ha="right")
    #plt.title('Maternal Health data set - boxplot')
    #plt.show()


    #Boxplot for each attribute
    classNames=["Low Risk", "Mid Risk", "High Risk"]

    fig_AGE = plt.boxplot(X[:,0])
    plt.xticks(range(1,2),["Age"])
    plt.ylabel("(Years)")
    plt.title('Age - boxplot')
    plt.show()

    fig_SP = plt.boxplot(X[:,1])
    plt.xticks(range(1,2),["Systolic Pressure"])
    plt.ylabel("(mmHg)")
    plt.title('Systolic Pressure - boxplot')
    plt.show()

    fig_DP = plt.boxplot(X[:,2])
    plt.xticks(range(1,2),["Diastolic Blood Pressure"])
    plt.ylabel("(mmHg)")
    plt.title("Diastolic BP - boxplot")
    plt.show()

    fig_BG = plt.boxplot(X[:,3])
    plt.xticks(range(1,2),["Blood Glucose"])
    plt.ylabel("(mmol/L)")
    plt.title("Blood Glucose - boxplot")
    plt.show()

    fig_BT = plt.boxplot(X[:,4])
    plt.ylabel("Temperature (\u00B0C)")
    plt.xticks(range(1,2),["Body Temperature"])
    plt.title("Body Temperature - boxplot")
    plt.show()

    fig_HR = plt.boxplot(X[:,5])
    plt.ylabel("(Beats per minute)")
    plt.xticks(range(1,2),["Heart Rate"])
    plt.title("Heart Rate - boxplot")
    plt.show()


    #Boxplot / classes / all-together
    plt.figure(figsize=(14,7))
    for c in range(3):
        plt.subplot(1,3,c+1)
        class_mask = (y==c+1) # binary mask to extract elements of class c

        
        plt.boxplot(X[class_mask,:6])
        plt.title('Class: {0}'.format(classNames[c]))
        plt.title('Class: '+classNames[c])
        plt.xticks(range(1,7), [a[:18] for a in attributeNames[:6]], rotation=45, ha="right")
        #y_up = X.max()+(X.max()-X.min())*0.1; y_down = X.min()-(X.max()-X.min())*0.1
        #plt.ylim(y_down, y_up)

    plt.show()

    #Boxplot - each attribute wth risk factor

    for j in range(6):
        plt.figure(figsize=(14,7))
        for c in range(3):
            plt.subplot(1,3,c+1)
            class_mask = (y==c+1) # binary mask to extract elements of class c

        
            plt.boxplot(X[class_mask,j])
            plt.title('Class: {0}'.format(classNames[c]))
            plt.title('Class: '+classNames[c])
            plt.xticks(range(1,2), [attributeNames[j]], rotation=45, ha="right")
            #y_up = X.max()+(X.max()-X.min())*0.1; y_down = X.min()-(X.max()-X.min())*0.1
            #plt.ylim(y_down, y_up)
        plt.show()
