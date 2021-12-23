"""
Created on Mon Dec  6 17:45:05 2021

@author: mxkep
"""
import sys
import utility as utl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pretty_confusion_matrix import plot_confusion_matrix_from_data as pretty_conf_matrix
#third party code to display confusion matrix


#old_stdout = sys.stdout
#log_file = open("output.log","w")
#sys.stdout = log_file

BitcoinHeistData = pd.read_csv("C:\\Users\\mxkep\\OneDrive\\HU\\530 Machine Learning\\project\\data\\BitcoinHeistData.csv")

     
# Take 10000 random samples from dataset
data = BitcoinHeistData.drop(['address', 'year', 'day'], axis = 1)

#data['ransomware'] = np.where(data['label'] == 'white', 'notRansomware', 'ransomware') # 0 for white, 1 for ransomware
data['ransomware'] = np.where(data['label'] == 'white', 0, 1) # 0 for notRansomware, 1 for ransomware

data_new = data[data.label != 'white']
data_new = data_new.append(data[data.label == 'white'].sample(n=41413, random_state=123))

data_new = data_new.drop(['label'], axis=1)
print(data_new.head())

target = data_new['ransomware']


# Split the data

from sklearn.model_selection import train_test_split
y = target
X = data_new.drop(['ransomware'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                                                   random_state = 25)


# Check Correlation

corr = X_train.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(X_train.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
plt.title("Correlation Plot")
ax.set_yticks(ticks)
ax.set_xticklabels(X_train.columns)
ax.set_yticklabels(X_train.columns)
plt.show()



########################################################################################
# Create Decision Tree
from sklearn import tree

model = tree.DecisionTreeClassifier()
model = model.fit(X_train, y_train)


# Feature Importance Plot

utl.pretty_importances_plot(
    model.feature_importances_, 
    [i for i in range(X_train.shape[1])],
    xlabel = 'Importance',
    ylabel = 'Feature',
    horizontal_label = 'Feature importance'
)





text_representation = tree.export_text(model)
with open("decision_tree.log", "w") as fout:
    fout.write(text_representation)

# Evaluate Decision Tree Model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

y_predict = model.predict(X_test)
y_pred_dt = y_predict


from sklearn.preprocessing import StandardScaler

#
# Standardize the data set
#
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#
# Plot confusion matrix


pretty_conf_matrix(
    y_test = y_test,
    predictions = y_pred_dt,
    string = "Decision Tree")

print("\nDecision Tree ")
utl.cv_model(X_train, y_train, model, "Decision Tree")
print("Decision Tree Accuracy Score: ", accuracy_score(y_test, y_predict) * 100)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict))
print("\n")



#####################################################################################

# Create Random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, random_state=123)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
y_pred_rf = y_predict

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# Plot confusion matrix

pretty_conf_matrix(
    y_test = y_test,
    predictions = y_pred_rf,
    string = "Random Forest")

print("Random Forest ")
utl.cv_model(X_train, y_train, model, "Random Forest")
print("Random Forest Accuracy Score: ", accuracy_score(y_test, y_predict) * 100)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict))
print("\n")




#######################################################################################
############## Train the Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)

# Evaluate model performance

y_predict = gnb.predict(X_test)
y_pred_nb = y_predict

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#
# Plot confusion matrix

pretty_conf_matrix(
    y_test = y_test,
    predictions = y_pred_nb,
    string = "Naive Bayes")

print("Naive Bayes ")
utl.cv_model(X_train, y_train, model, "Naive Bayes")
print("Naive Bayes Classifier Accuracy Score: ", accuracy_score(y_test, y_predict)*100)
print("Confusion Matrix: \n", confusion_matrix(y_test, y_predict))
print("\n")

#######################################################################################



##################### Design SVM poly kernel  #####################


from sklearn import svm

# Take N samples
data_svm = data_new.sample(n=6000, random_state=987 )


data_svm = data_svm.drop(['count', 'looped'], axis = 1)

target_svm = data_svm['ransomware']

y = target_svm
X = data_svm.drop(['ransomware'], axis = 1)

# Save for later to print in table
y_test_1 = y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,
                                                   random_state = 25)


# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)



clf = svm.SVC(kernel='poly')


clf.fit(X_train, y_train)


y_predict = clf.predict(X_test)
y_pred_poly = y_predict


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#
# Plot confusion matrix

pretty_conf_matrix(
    y_test = y_test,
    predictions = y_pred_poly,
    string = "SVM (polynomial)")


print("SVM (polynomial) ")
utl.cv_model(X_train, y_train, clf, "SVM (polynomial)")
print("SVM (polynomial)  Accuracy Score: ", accuracy_score(y_test, y_pred_poly)*100)
print("SVM (polynomial) Confusion Matrix: \n", confusion_matrix(y_test, y_pred_poly))
print("\n")



##################### Design SVM rbf kernel #####################

clf = svm.SVC(kernel='rbf')


clf.fit(X_train, y_train)


y_predict = clf.predict(X_test)
y_pred_rbf = y_predict


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#
# Plot confusion matrix

pretty_conf_matrix(
    y_test = y_test,
    predictions = y_pred_rbf,
    string = "SVM (RBF)")

print("SVM (RBF) ")
utl.cv_model(X_train, y_train, clf, "SVM (RBF)")
print("SVM (RBF) Accuracy Score: ", accuracy_score(y_test, y_pred_rbf)*100)
print("SVM (RBF)  Confusion Matrix: \n", confusion_matrix(y_test, y_pred_rbf))
print("\n")



################## Print Results ##################

utl.print_table(y_test_1, y_pred_dt, y_pred_rf, y_pred_nb)
utl.print_table_svm(y_test, y_pred_poly, y_pred_rbf, "SVM (polynomial)", "SVM (RBF)")



#sys.stdout = old_stdout
#log_file.close()
#%%















