import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,2:4]
y = dataset.iloc[:,4]

#splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

def Classification(classifier,type):
    print(type)
    classifier.fit(X_train,y_train)

    #predicting the Test set result
    y_pred = classifier.predict(X_test)

    #making the confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,y_pred)
    print("confusion matrix",cm)

    #visualising the Training set result
    from matplotlib.colors import ListedColormap
    X_set,y_set = X_train,y_train
    X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step=0.01),
                        np.arange(start=X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step=0.01))
    plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75,cmap = ListedColormap(('red','green')))
    plt.xlim(X1.min(),X1.max())
    plt.ylim(X2.min(),X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                    c = ListedColormap(('red','green'))(i),label = j)
    plt.title(type+'(Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()

    #visualising the Test Set result
    from matplotlib.colors import ListedColormap
    X_set,y_set = X_test,y_test
    X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop = X_set[:,0].max()+1,step=0.01),
                        np.arange(start=X_set[:,1].min()-1,stop = X_set[:,1].max()+1,step=0.01))
    plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.75,cmap = ListedColormap(('red','green')))
    plt.xlim(X1.min(),X1.max())
    plt.ylim(X2.min(),X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                    c = ListedColormap(('red','green'))(i),label = j)
    plt.title(type+'(Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()


#-------------------------------------------------------------------------------------------------

#fitting logistic regression to the dataset
from sklearn.linear_model import LogisticRegression
Classification(LogisticRegression(random_state = 0),'Logistic Regression')

#-------------------------------------------------------------------------------------------------

#fitting k-NN to the dataset
from sklearn.neighbors import KNeighborsClassifier
Classification(KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2),'K-NN')

#-------------------------------------------------------------------------------------------------

#fitting svm to the dataset
from sklearn.svm import SVC
Classification(SVC(kernel='linear',random_state = 0),'SVM')

#-------------------------------------------------------------------------------------------------

#fitting svm kernel to the dataset
from sklearn.svm import SVC
Classification(SVC(kernel='rbf',random_state = 0),'SVM KERNEL')

#-------------------------------------------------------------------------------------------------

#fitting Naive Bayes to the dataset
from sklearn.naive_bayes import GaussianNB
Classification(GaussianNB(),'NAIVE BAYES')

#-------------------------------------------------------------------------------------------------

#fitting Decision Tree to the dataset
from sklearn.tree import DecisionTreeClassifier
Classification(DecisionTreeClassifier(),'DECISION TREE')

#-------------------------------------------------------------------------------------------------
