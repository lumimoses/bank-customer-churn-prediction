#!/usr/bin/env python
# coding: utf-8

# <a id=0></a>
# ## [Introduction](#0)
# Bank customer churn prediction using CNN
# 
# ## Workflow
# 
# 1. [Data collection and initial analysis](#1)
# 2. [EDA](#2)
# 3. [Data-preprocessing](#3)
# 4. [Model building](#4)
# 5. [Test data predictions](#5)

# <a id=1></a> 
# ## 1. Data collection and initial analysis

# In[1]:


#Import required libraries

#Computing libraries
import pandas as pd
import numpy as np

#Visualizations library
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


#Model building libraries
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

#Deep learning libraries
import tensorflow as tf

#DL library to use forward and backward propogation
from tensorflow.keras.models import Sequential

#DL library to build input/hidden/output layers
from tensorflow.keras.layers import Dense

#DL library to prevent overfitting
from tensorflow.keras.layers import Dropout

# DL libraries for convolutional layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape

#DL library to use activation function
from tensorflow.keras.layers import LeakyReLU, ReLU

#DL library to use optimizer
from tensorflow.keras.optimizers import Adam


#Performance metrics
from sklearn.metrics import (confusion_matrix,accuracy_score, 
                             classification_report, roc_curve,
                             roc_auc_score)

#import library used for counting the number of observations
from collections import Counter

#import library to perform resampling
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Check the installed tensorflow version
print(tf.__version__)


# In[3]:


#dataset link='https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers'

#load the dataset
df_data = pd.read_csv('/kaggle/input/churn-for-bank-customers/churn.csv').drop(['RowNumber'], axis = 1)
df_data.head()


# In[4]:


df_data.info()


# ### Feature Description:
# 
# * RowNumber—corresponds to the record (row) number and has no effect on the output.
# * CustomerId—contains random values and has no effect on customer leaving the bank.
# * Surname—the surname of a customer has no impact on their decision to leave the bank.
# * CreditScore—can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
# * Geography—a customer’s location can affect their decision to leave the bank.
# * Gender—it’s interesting to explore whether gender plays a role in a customer leaving the bank.
# * Age—this is certainly relevant, since older customers are less likely to leave their bank than younger ones.
# * Tenure—refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
# * Balance—also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
# * NumOfProducts—refers to the number of products that a customer has purchased through the bank.
# * HasCrCard—denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
# * IsActiveMember—active customers are less likely to leave the bank.
# * EstimatedSalary—as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
# * Exited—whether or not the customer left the bank.

# In[5]:


#Check for missing values
df_data.isnull().sum()


# <a id=2></a> 
# ## 2. EDA

# In[6]:


# Target variable distribution
total = float(len(df_data))
isbalanced = df_data.Exited.value_counts() / total
print(isbalanced.apply(lambda x: f'{100 * x:.2f}%'))

ax = sns.countplot(x='Exited', data=df_data)
for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='right', va='bottom', fontsize = 12)


# ## Correlation Analysis

# In[7]:


corrmat=df_data.corr()
corrmat


# In[8]:


plt.figure(figsize=(10,8))
sns.heatmap(corrmat, square=True, annot=True)


# In[9]:


# Specify the pairs of variables with high correlation
high_corr_pairs = [('Age', 'Exited'), ('NumOfProducts', 'Balance')]

# Create scatterplots for each pair
plt.figure(figsize=(12, 6))
for idx, (var1, var2) in enumerate(high_corr_pairs, 1):
    plt.subplot(1, len(high_corr_pairs), idx)
    plt.scatter(df_data[var1], df_data[var2], alpha=0.5)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(f'{var1} vs {var2}')

plt.tight_layout()
plt.show()


# In[10]:


def plot_categorical_feature(col, churn_data=df_data):
    table = pd.crosstab(index=churn_data[col], columns=churn_data['Exited'], margins=True, normalize='all') * 100

    # Rename the columns
    table.columns = ['Non-Churned', 'Churned', 'Total']

    # Convert percentages to fractions of 100%
    table['Non-Churned'] = table['Non-Churned'].apply(lambda x: f'{x:.2f}%')
    table['Churned'] = table['Churned'].apply(lambda x: f'{x:.2f}%')
    table['Total'] = table['Total'].apply(lambda x: f'{x:.2f}%')

    print(f'Bivariate Percentage Table: {col} vs. Exited')
    print(table)
    print()

    # Plot univariate pie chart
    plt.figure(figsize=(10, 5))

    """plt.subplot(1, 2, 1)
    label = churn_data[col].value_counts().index
    label_count = churn_data[col].value_counts().values

    plt.pie(x=label_count, labels=label, autopct='%1.1f%%', shadow=True, radius=1)
    plt.title(f'Univariate Pie Chart: {col} Distribution')

    # Plot grouped bar chart
    plt.subplot(1, 2, 2)
    """
    total = float(len(churn_data))

    ax = sns.countplot(data=churn_data, x=col, hue='Exited')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'Distribution of {col} for Churned vs. Non-Churned Customers')

    for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_height() / total)
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='right', va='bottom', fontsize=14)

    plt.legend(title='Exited')
    plt.tight_layout()
    plt.show()


# In[11]:


# Example usage
categorical_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember', 'NumOfProducts']
for col in categorical_cols:
    plot_categorical_feature(col)


# In[12]:


def plot_numerical_features(col, churn_data = df_data):
    grouped_stats = churn_data.groupby('Exited')[numerical_cols].describe()
    print(f"Variable: {col}")
    print(grouped_stats[col])
    print()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=churn_data, x='Exited', y=col)
    plt.xlabel('Exited (Churned)')
    plt.ylabel(col)
    plt.title(f'Distribution of {col} for Churned vs. Non-Churned Customers (Boxplot)')
    plt.show()


# In[13]:


# Example usage
numerical_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
for col in numerical_cols:
    plot_numerical_features(col)


# In[14]:


# Define the relationships and corresponding variables
relationships = {
    'Age vs Gender': ['Gender', 'Age'],
    'Tenure vs IsActiveMember': ['IsActiveMember', 'Tenure'],
    'Balance vs IsActiveMember': ['IsActiveMember', 'Balance'],
    'CreditScore vs IsActiveMember': ['IsActiveMember', 'CreditScore'],
    'NumOfProducts vs HasCrCard': ['HasCrCard', 'NumOfProducts'],
    'NumOfProducts vs Balance': ['NumOfProducts', 'Balance'],
}

# Create violin plots for each relationship
for relationship, variables in relationships.items():
    """
    cumulative_table = pd.crosstab(index=df_data[variables[0]], columns=[df_data[variables[1]], df_data['Exited']], normalize='all', margins=True) * 100

    
    # Rename the columns
    cumulative_table.columns = pd.MultiIndex.from_tuples([(f'{col[0]} (Non-Churned)', col[1]) for col in cumulative_table.columns])
    
    # Calculate cumulative frequencies
    cumulative_table = cumulative_table.cumsum(axis=0)
    
    # Convert percentages to fractions of 100%
    cumulative_table = cumulative_table.applymap(lambda x: f'{x:.2f}%')
    
    # Display the table
    print(f'Multivariate Cumulative Frequency Table: {relationship}')
    display(cumulative_table)
    """
    
    
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df_data, x=variables[0], y=variables[1], hue='Exited', split=True)
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
    plt.title(f'{relationship} against Exited (Violin Plot)')
    plt.legend(title='Exited', loc='upper right')
    plt.show()


# <a id=3></a> 
# ## 3. Data-preprocessing

# In[15]:


#drop features which are not required
df_data.drop(['CustomerId', 'Surname'], axis=1, inplace=True)


# In[16]:


#Onehot encode categorical features
geog = pd.get_dummies(df_data['Geography'], drop_first=True)
gen = pd.get_dummies(df_data['Gender'], drop_first=True)


# In[17]:


geog.head()


# In[18]:


gen.head()


# In[19]:


#Concat the encoded features to the main dataframe
df_data = pd.concat([df_data,gen,geog],axis=1)

#Drop the old categorical features
df_data.drop(['Geography', 'Gender'], axis=1, inplace=True)


# In[20]:


df_data.head()


# <a id=4></a> 
# ## 4. Model building

# In[21]:


#Seperate independent and dependent features
X = df_data.loc[:, df_data.columns!='Exited']
y = df_data['Exited']

X.shape, y.shape


# In[22]:


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=0)
# summarize
print('Train', X_train.shape, y_train.shape)
print('Test', X_valid.shape, y_valid.shape)


# In[23]:


# Feature Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_valid_sc = scaler.transform(X_valid)


# In[24]:


def naive_model(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
    return classifier

def tuned_model(classifier, param_grid, X_train, y_train):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(classifier, param_grid, cv=cv, scoring='roc_auc')
    grid.fit(X_train, y_train)
    return grid

def evaluate_model(classifier, X_valid, y_valid):
    if isinstance(classifier, GridSearchCV):
        best_classifier = classifier.best_estimator_
    else:
        best_classifier = classifier
    
    prediction = best_classifier.predict(X_valid)
    
    # Convert predictions to 0 or 1 based on threshold
    if isinstance(best_classifier, Sequential):  # Check if neural network
        prediction = (prediction >= 0.5).astype(int)
        
    roc_auc = roc_auc_score(y_valid, prediction)
    accuracy = accuracy_score(y_valid, prediction)
    
    print("Accuracy: {:.2%}".format(accuracy))
    
    print("ROC AUC Score: {:.2%}".format(roc_auc))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_valid, prediction)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.subplot(1, 2, 2)
    # Plot confusion matrix
    cm = confusion_matrix(y_valid, prediction)
    names = ['True Neg','False Pos','False Neg','True Pos']
    counts = [value for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cm,annot = labels,fmt ='', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Classification report
    class_report = classification_report(y_valid, prediction)
    print("Classification Report:\n", class_report)


# 1. Logistic Regression

# In[25]:


# Naive Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state=0)
naive_lr_model = naive_model(classifier_lr, X_train_sc, y_train)


# In[26]:


# Hyperparameter Tuned Logistic Regression
param_grid_lr = {'C': [0.01, 0.1, 1, 10],
                 'penalty': ['l1', 'l2']}
grid_lr = tuned_model(classifier_lr, param_grid_lr, X_train_sc, y_train)
tuned_lr_model = grid_lr.best_estimator_


# In[27]:


# Evaluate Naive Logistic Regression
evaluate_model(naive_lr_model, X_valid_sc, y_valid)


# In[28]:


# Evaluate Tuned Logistic Regression
evaluate_model(grid_lr, X_valid_sc, y_valid)


# 2. Support Vector Classifier

# In[29]:


# Support Vector Classifier
from sklearn.svm import SVC


# In[30]:


# Naive Support Vector Classifier
classifier_svc = SVC(kernel='linear', C=1)
naive_svc_model = naive_model(classifier_svc, X_train_sc, y_train)


# In[31]:


# Hyperparameter Tuned Support Vector Classifier
param_grid_svc = {'C': [0.01, 0.1, 1, 10],
                  'kernel': ['linear', 'rbf']}
grid_svc = tuned_model(classifier_svc, param_grid_svc, X_train_sc, y_train)
tuned_svc_model = grid_svc.best_estimator_


# In[32]:


# Evaluate Naive Support Vector Classifier
evaluate_model(naive_svc_model, X_valid_sc, y_valid)


# In[33]:


# Evaluate Tuned Support Vector Classifier
evaluate_model(grid_svc, X_valid_sc, y_valid)


# 3. Decision Tree Classifier

# In[34]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier


# In[35]:


# Naive Decision Tree Classifier
classifier_dt = DecisionTreeClassifier(random_state=0, max_depth=4, min_samples_leaf=1)
naive_dt_model = naive_model(classifier_dt, X_train_sc, y_train)


# In[36]:


# Hyperparameter Tuned Decision Tree Classifier
param_grid_dt = {'max_depth': [3, 4, 5],
                 'min_samples_split': [2, 5, 10]}
grid_dt = tuned_model(classifier_dt, param_grid_dt, X_train_sc, y_train)
tuned_dt_model = grid_dt.best_estimator_


# In[37]:


# Evaluate Naive Decision Tree Classifier
evaluate_model(naive_dt_model, X_valid_sc, y_valid)


# In[38]:


# Evaluate Tuned Decision Tree Classifier
evaluate_model(grid_dt, X_valid_sc, y_valid)


# 
# 4. Random Forest Clasifier

# In[39]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


# In[40]:


# Naive Random Forest Classifier
classifier_rf = RandomForestClassifier(max_depth=4, random_state=0)
naive_rf_model = naive_model(classifier_rf, X_train_sc, y_train)


# In[41]:


# Hyperparameter Tuned Random Forest Classifier
param_grid_rf = {'max_depth': [3, 4, 5],
                 'n_estimators': [100, 150, 200]}
grid_rf = tuned_model(classifier_rf, param_grid_rf, X_train_sc, y_train)
tuned_rf_model = grid_rf.best_estimator_


# In[42]:


# Evaluate Naive Random Forest Classifier
evaluate_model(naive_rf_model, X_valid_sc, y_valid)


# In[43]:


# Evaluate Tuned Random Forest Classifier
evaluate_model(grid_rf, X_valid_sc, y_valid)


# 5. K-Nearest Neigbours Classifier

# In[44]:


# K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier


# In[45]:


# Naive K-Nearest Neighbors Classifier
classifier_knn = KNeighborsClassifier(n_neighbors=3)
naive_knn_model = naive_model(classifier_knn, X_train_sc, y_train)


# In[46]:


# Hyperparameter Tuned K-Nearest Neighbors Classifier
param_grid_knn = {'n_neighbors': [3, 5, 7],
                  'weights': ['uniform', 'distance']}
grid_knn = tuned_model(classifier_knn, param_grid_knn, X_train_sc, y_train)
tuned_knn_model = grid_knn.best_estimator_


# In[47]:


# Evaluate Naive K-Nearest Neighbors Classifier
evaluate_model(naive_knn_model, X_valid_sc, y_valid)


# In[48]:


# Evaluate Tuned K-Nearest Neighbors Classifier
evaluate_model(grid_knn, X_valid_sc, y_valid)


# Feed Forward Neural Networks

# In[49]:


def fit_model(classifier, X_train, y_train, early_stop):
    model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=1000, callbacks=early_stop)
    return model_history

def plot_accuracy_loss(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()


# * The activation function helps to determine the output of a neural network. These type of functions are attached to each neuron in the network and determines whether it should be activated or not, based on whether each neuron's input is relevant for the model's prediction.
# * The sigmoid function is used on the output layer for a binary classification problem as it gives an output of either 0 or 1.
# * The ReLU (Rectified Linear Unit) function is used as an activation function on the hidden layers. It uses the equation max(0,z) which means it generates the maximum value between 0 and z
# * The dropout function is used to prevent overfitting. It deactivates some of the neurons in the given hidden layer.

# In[50]:


#Initialise ANN
classifier = Sequential()

#Add input layer
classifier.add(Dense(units=11,activation='relu'))

#Add first hidden layer
classifier.add(Dense(units=7, activation='relu'))
classifier.add(Dropout(0.2))

#Add second hidden layer
classifier.add(Dense(units=7, activation='relu'))
classifier.add(Dropout(0.2))

#Add third hidden layer
classifier.add(Dense(units=7, activation='relu'))
classifier.add(Dropout(0.2))

#Add output layer
classifier.add(Dense(units=1, activation='sigmoid'))


# * Optimizer is used to update the weights and bias during the back-propagation. The Adam optimizer uses noise smoothening and adaptive learning rate.
# * The binary cross-entropy loss function is used for binary classification problems. Here y is the actual value and ŷ is the predicted value.

# In[51]:


#optimizer
opt = Adam(learning_rate=0.01)

#compile
classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# Early stopping is used to stop the Neural Network if there is no significant improvement in the model's accuracy.

# In[52]:


#Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
)

#monitor: Quantity to be monitored.

#min_delta: Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.

#patience: Number of epochs with no improvement after which training will be stopped.

#verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action.

#mode: One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing; in "auto" mode, the direction is automatically inferred from the name of the monitored quantity.

#baseline: Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline.

#restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used. An epoch will be restored regardless of the performance relative to the baseline. If no epoch improves on baseline, training will run for patience epochs and restore weights from the best epoch in that set.


# In[53]:


model_history = fit_model(classifier, X_train_sc, y_train, early_stop)


# In[54]:


plot_accuracy_loss(model_history)


# <a id=5></a> 
# ## 5. Test data Prediction

# In[55]:


evaluate_model(classifier, X_valid_sc, y_valid)


# Convolutional Neural Networks

# In[56]:


# Reshape the data
X_train_reshaped = X_train_sc.reshape(-1, 11, 1, 1)
X_valid_reshaped = X_valid_sc.reshape(-1, 11, 1, 1)


# In[57]:


# Initialize CNN
classifier = Sequential()

# Add Convolutional layer
classifier.add(Conv2D(32, (1, 1), activation='relu', input_shape=(11, 1, 1)))


# Add Max Pooling layer
classifier.add(MaxPooling2D(pool_size=(1, 1)))

# Flatten the layer
classifier.add(Flatten())

# Add hidden layer with rectifier activation function
classifier.add(Dense(units=7, activation='relu'))
classifier.add(Dropout(0.2))

# Add output layer with sigmoid activation function
classifier.add(Dense(units=1, activation='sigmoid'))


# In[58]:


classifier.summary()


# In[59]:


# Compile the model
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# In[60]:


model_history = fit_model(classifier, X_train_reshaped, y_train, early_stop)


# In[61]:


plot_accuracy_loss(model_history)


# In[62]:


evaluate_model(classifier, X_valid_reshaped, y_valid)


# In[63]:


# Define the hyperparameters and their possible values
param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [100, 200, 300],
    'optimizer': ['adam', 'rmsprop'],
    'hidden_layers': [1, 2, 3],
    'neurons_per_layer': [16, 32, 64],
    'dropout_rate': [0.2, 0.3, 0.4],
}

# Create a function to build and compile the model
def build_model(optimizer, hidden_layers, neurons_per_layer, dropout_rate):
    model = Sequential()
    model.add(Dense(neurons_per_layer, activation='relu', input_shape=(X_train_sc.shape[1],)))
    for _ in range(hidden_layers):
        model.add(Dense(neurons_per_layer, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier wrapper for Scikit-Learn compatibility
keras_classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model, verbose=0)


# In[ ]:


# Perform grid search
grid_fnn = tuned_model(keras_classifier, param_grid, X_train_sc, y_train)
tuned_fnn_model = grid_fnn.best_estimator_


# In[ ]:


tuned_fnn_model.summary()


# In[ ]:


model_history = fit_model(tuned_fnn_model, X_train_sc, y_train, early_stop)


# In[ ]:


plot_accuracy-loss(model_history)


# In[ ]:


# Evaluate the best model
evaluate_model(grid_fnn, X_valid_sc, y_valid)


# In[ ]:


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Define the hyperparameters and their possible values
param_grid_cnn = {
    'batch_size': [16, 32, 64],
    'epochs': [100, 200, 300],
    'optimizer': ['adam', 'rmsprop'],
    'hidden_layers': [1, 2, 3],
    'neurons_per_layer': [16, 32, 64],
    'dropout_rate': [0.2, 0.3, 0.4],
}

# Create a function to build and compile the CNN model
def build_cnn_model(optimizer, hidden_layers, neurons_per_layer, dropout_rate):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(11, 1, 1)))
    cnn_model.add(MaxPooling2D(pool_size=(1, 1)))
    cnn_model.add(Flatten())
    
    for _ in range(hidden_layers):
        cnn_model.add(Dense(neurons_per_layer, activation='relu'))
        cnn_model.add(Dropout(dropout_rate))
    
    cnn_model.add(Dense(1, activation='sigmoid'))
    cnn_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return cnn_model

# Create a KerasClassifier wrapper for Scikit-Learn compatibility
keras_cnn_classifier = KerasClassifier(build_fn=build_cnn_model, verbose=0)


# In[ ]:


# Perform grid search
grid_cnn = tuned_model(keras_cnn_classifier, param_grid_cnn, X_train_reshaped, y_train)
tuned_cnn_model = grid_cnn.best_estimator_


# In[ ]:


tuned_fnn_model.summary()


# In[ ]:


model_history = fit_model(tuned_cnn_model, X_train_sc, y_train, early_stop)


# In[ ]:


plot_accuracy-loss(model_history)


# In[ ]:


# Evaluate the best model
evaluate_model(grid_cnn, X_valid_reshaped, y_valid)


# In[ ]:




