import numpy as np
import pandas as pd
from sklearn import model_selection
import torch
import scipy.stats as st
from toolbox_02450 import rlr_validate


def train_neural_net(model, loss_fn, X, y,
                     n_replicates=3, max_iter = 10000, tolerance=1e-6):
    """
    Train a neural network with PyTorch based on a training set consisting of
    observations X and class y. The model and loss_fn inputs define the
    architecture to train and the cost-function update the weights based on,
    respectively.
    
    Usage:
        Assuming loaded dataset (X,y) has been split into a training and 
        test set called (X_train, y_train) and (X_test, y_test), and
        that the dataset has been cast into PyTorch tensors using e.g.:
            X_train = torch.tensor(X_train, dtype=torch.float)
        Here illustrating a binary classification example based on e.g.
        M=2 features with H=2 hidden units:
    
        >>> # Define the overall architechture to use
        >>> model = lambda: torch.nn.Sequential( 
                    torch.nn.Linear(M, H),  # M features to H hiden units
                    torch.nn.Tanh(),        # 1st transfer function
                    torch.nn.Linear(H, 1),  # H hidden units to 1 output neuron
                    torch.nn.Sigmoid()      # final tranfer function
                    ) 
        >>> loss_fn = torch.nn.BCELoss() # define loss to use
        >>> net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=3)
        >>> y_test_est = net(X_test) # predictions of network on test set
        >>> # To optain "hard" class predictions, threshold the y_test_est
        >>> See exercise ex8_2_2.py for indepth example.
        
        For multi-class with C classes, we need to change this model to e.g.:
        >>> model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, H), #M features to H hiden units
                            torch.nn.ReLU(), # 1st transfer function
                            torch.nn.Linear(H, C), # H hidden units to C classes
                            torch.nn.Softmax(dim=1) # final tranfer function
                            )
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        
        And the final class prediction is based on the argmax of the output
        nodes:
        >>> y_class = torch.max(y_test_est, dim=1)[1]
        
    Args:
        model:          A function handle to make a torch.nn.Sequential.
        loss_fn:        A torch.nn-loss, e.g.  torch.nn.BCELoss() for binary 
                        binary classification, torch.nn.CrossEntropyLoss() for
                        multiclass classification, or torch.nn.MSELoss() for
                        regression (see https://pytorch.org/docs/stable/nn.html#loss-functions)
        n_replicates:   An integer specifying number of replicates to train,
                        the neural network with the lowest loss is returned.
        max_iter:       An integer specifying the maximum number of iterations
                        to do (default 10000).
        tolerenace:     A float describing the tolerance/convergence criterion
                        for minimum relative change in loss (default 1e-6)
                        
        
    Returns:
        A list of three elements:
            best_net:       A trained torch.nn.Sequential that had the lowest 
                            loss of the trained replicates
            final_loss:     An float specifying the loss of best performing net
            learning_curve: A list containing the learning curve of the best net.
    
    """
    
    # Specify maximum number of iterations for training
    logging_frequency = 1000 # display the loss every 1000th iteration
    best_final_loss = 1e100
    for r in range(n_replicates):
        #print('\n\tReplicate: {}/{}'.format(r+1, n_replicates))
        # Make a new net (calling model() makes a new initialization of weights) 
        net = model()
        
        # initialize weights based on limits that scale with number of in- and
        # outputs to the layer, increasing the chance that we converge to 
        # a good solution
        torch.nn.init.xavier_uniform_(net[0].weight)
        torch.nn.init.xavier_uniform_(net[2].weight)
                     
        # We can optimize the weights by means of stochastic gradient descent
        # The learning rate, lr, can be adjusted if training doesn't perform as
        # intended try reducing the lr. If the learning curve hasn't converged
        # (i.e. "flattend out"), you can try try increasing the maximum number of
        # iterations, but also potentially increasing the learning rate:
        #optimizer = torch.optim.SGD(net.parameters(), lr = 5e-3)
        
        # A more complicated optimizer is the Adam-algortihm, which is an extension
        # of SGD to adaptively change the learing rate, which is widely used:
        optimizer = torch.optim.Adam(net.parameters())
        
        # Train the network while displaying and storing the loss
        #print('\t\t{}\t{}\t\t\t{}'.format('Iter', 'Loss','Rel. loss'))
        learning_curve = [] # setup storage for loss at each step
        old_loss = 1e6
        for i in range(max_iter):      
            y_est = net(X) # forward pass, predict labels on training set
            y_est = torch.squeeze(y_est)
            loss = loss_fn(y_est, y) # determine loss
            loss_value = loss.data.numpy() #get numpy array instead of tensor
            learning_curve.append(loss_value) # record loss for later display
            
            # Convergence check, see if the percentual loss decrease is within
            # tolerance:
            p_delta_loss = np.abs(loss_value-old_loss)/old_loss
            if p_delta_loss < tolerance: break
            old_loss = loss_value
            
            # display loss with some frequency:
            if (i != 0) & ((i+1) % logging_frequency == 0):
                print_str = '\t\t' + str(i+1) + '\t' + str(loss_value) + '\t' + str(p_delta_loss)
                #print(print_str)
            # do backpropagation of loss and optimize weights 
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
        
        if loss_value < best_final_loss: 
            best_net = net
            best_final_loss = loss_value
            best_learning_curve = learning_curve
        
    # Return the best curve along with its final loss and learing curve
    return best_net, best_final_loss, best_learning_curve

def ttest_twomodels(y_true, yhatA, yhatB, alpha=0.05, loss_norm_p=1):
    zA = np.abs(y_true - yhatA) ** loss_norm_p
    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    zB = np.abs(y_true - yhatB) ** loss_norm_p

    z = zA - zB
    CI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
    return np.mean(z), CI, p

#Load the data in pandas dataframe
df = pd.read_csv("../Data/MaternalDataset.csv")

raw_data = df.values
classes = raw_data[:-2,-1]
classes = list(set(classes))

classNames = {"low risk":1,"mid risk":2, "high risk":3}
y = np.asarray([classNames[value] for value in raw_data[:,-1]])

# use features selected from regression a part
attributeNames = ["Systolic BP", "Blood Glucose", "Body Temperature", "Heart Rate", "Risk Level"]
X = raw_data[:,[1,3,4,5]]
N,M = X.shape

X = X.astype(float)
y = y.astype(float)

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

h = 50
lambdas = [10]

K = 10
max_iter = 10000
CV = model_selection.KFold(K, shuffle=True)
ANN_est = []
lr_est = []
base_est = []
y_true = []
k=0
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_rlr = np.empty((M,K))
Error_test_rlr = np.empty((K,1))
regression_table = np.array([])
# Using the generalization error formula for k-fold cross validation
alpha = 0.05
loss_norm_p  = 2
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    y_true = np.append(y_true, y_test)

    ## ANN
    # Parameters for neural network classifier
    n_hidden_units = h          # number of hidden units
    n_replicates = 1            # number of networks trained in each k-fold
    
    # Define the model
    ANNmodel = lambda: torch.nn.Sequential(
                        torch.nn.Linear(X.shape[1], n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(ANNmodel,
                                                       loss_fn,
                                                       X=torch.Tensor(X_train),
                                                       y=torch.Tensor(y_train),
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
        
    # Determine estimated class labels for test set
    y_test_est = net(torch.Tensor(X_test)).detach().numpy().squeeze()
    ANN_est = np.append(ANN_est, y_test_est)
    
    ## Linear Regression
#    Xty = X_train.T @ y_train
#    XtX = X_train.T @ X_train
#
#    # Estimate weights for the optimal value of lambda, on entire training set
#    lambdaI = l * np.eye(X_train.shape[1])
#    lambdaI[0,0] = 0 # Do no regularize the bias term
#    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
#    
#    # Compute mean squared error with regularization with optimal lambda
#    y_test_est = X_test @ w_rlr
#    lr_est = np.append(lr_est, y_test_est)
    
    ## regularized linear model
    #internal_cross_validation set to 30 in order to fulfill the Central limit theorem   
    internal_cross_validation = 30
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute estmated y with regularization with optimal lambda
    y_test_est = X_test @ w_rlr[:,k]
    lr_est = np.append(lr_est, y_test_est)
    
    ## Baseline
    y_test_est = y_train.mean()
    base_est = np.append(base_est, [y_test_est]*len(y_test))
    
    
    
    print(k+1, "fold")

    print("\nANN vs baseline")
    [zmean_ANN_vs_base, CI_ANN_vs_base, p_ANN_vs_base] = ttest_twomodels(y_true, ANN_est, base_est, alpha, loss_norm_p)
    print(CI_ANN_vs_base,p_ANN_vs_base,sep="\n")
    print("\nLinear regression vs baseline")
    [zmean_lr_vs_base, CI_lr_vs_base, p_lr_vs_base] = ttest_twomodels(y_true, lr_est, base_est, alpha, loss_norm_p)
    print(CI_lr_vs_base,p_lr_vs_base,sep="\n")
    print("\nANN vs Linear regression")
    [zmean_ANN_vs_lr, CI_ANN_vs_lr, p_ANN_vs_lr] = ttest_twomodels(y_true, ANN_est, lr_est, alpha, loss_norm_p)
    print(CI_ANN_vs_lr,p_ANN_vs_lr,sep="\n")
    
    k += 1
print("Test end")