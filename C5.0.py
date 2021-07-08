#Importing packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# import data
wine = pd.read_csv("./winequality-white.csv")
print("Printing the first 20 tuples of the wine dataset:\n",wine.head(20),"\n")

# Data processing
x=wine.iloc[:,0:11] #features  #Selecting All tuples with starting 11 attributes
y=wine['quality']   # Quality Labels


print("Features:\n",x,"\n")
print("Quality Labels:\n",y,"\n")

print("Different Quality labels present in the wine dataset:",wine['quality'].unique(),"\n")

print("Labels counts:\n",wine.quality.value_counts(),"\n")

colnames = list(wine.columns)
print("Column names:\n",colnames,"\n")


# Splitting data into training and test data set
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


print("Shape of train and test datasets:\n", x_train, x_test, y_train, y_test,"\n")


#### Building C5.0 Decision Tree Classifier 

'''
C5.0 decision tree can be easily produced when we set the criterion = "entropy"
in the DecisionTreeClassifier 
'''

model = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, max_leaf_nodes=7) 
model.fit(x_train,y_train)


#Plotting the decision tree

#tree.plot_tree(model) # This will produce the default version of decision tree
#plt.show()

attributes= colnames[:11]
quality_values=['3', '4','5','6','7','8','9']

#quality_values= list(y.unique()) # This will not work because it will generate integer values type list and not string type

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,3), dpi=370)
tree.plot_tree(model,
               feature_names = attributes, 
               class_names = quality_values,
               filled = True);
plt.show()


# Predicting on test data
preds = model.predict(x_test) # predicting on test data set 

print("Printing Quality labels test data value counts\n:",y_test.value_counts(),"\n")

# Result Evaluation
print("Classicfication report:\n",classification_report(y_test,preds))

print("Confusion Matrix:\n",confusion_matrix(y_test,preds))

print("\nAccuracy:",accuracy_score(y_test,preds))

