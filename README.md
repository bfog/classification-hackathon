# Classification Hackathon
A 2 week hackathon conducted to develop a classification learning agent which is able to classify unknown platforms

## GridSearchCV results

### Configuration:
* Route used: Route 1
* Window size: 300
* Step size: 100
* Test split: 0.2
* Use unknowns: Yes
Features used: Velocity, Altitude, Heading

### Base scores:
* Gaussian NB: 11.78%
* SVM: 39.51%
* Decision Tree: 70.92%
* NCA & KNN: 40.74%
* Random Forest: 81.96%

GridSearchCV was used for Random Forest only, as it has the highest base scoring. The aim is to improve this score by finding the optimal parameters.
Random Forest base parameters were set to the default values: n_estimators: 100, random_state: 1

# Run 1
Execution Time: 2562s

UsedParameters: **[{'n_estimators': [10, 100, 1000], 'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2'], 'max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]**

Best params: **{'criterion': 'entropy', 'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 1000}**

Best score: **0.8146210476639923**

#### Classification Report:
              precision    recall  f1-score   support
           0       0.88      0.84      0.86        45
           1       0.87      0.83      0.85        41
           2       0.75      0.75      0.75        48
           3       0.85      0.85      0.85        26
           4       0.89      0.55      0.68        29
           5       0.77      0.96      0.86        71
           6       1.00      0.91      0.96        35
           7       0.91      0.71      0.80        14
           8       0.70      0.50      0.58        28
           9       0.80      0.46      0.58        35
          10       1.00      0.87      0.93        15
          11       0.97      0.99      0.98        92
          12       0.93      0.76      0.84        34
          13       0.67      0.67      0.67         6
          14       0.87      0.95      0.91        43
          15       0.88      0.82      0.85        28
          16       0.88      0.90      0.89        51
          17       0.75      0.88      0.81       174

    accuracy                           0.84       815
    macro avg      0.85      0.79      0.81       815
    weighted avg   0.84      0.84      0.83       815

Execute Random Forest with best parameters: **84.42%**

# Run 2
Execution Time: 159s

UsedParameters: **[{'n_estimators': [140, 150, 160], 'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2']}]**

Best params: **{'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 160}**

Best score: **0.8167663992449269**

#### Classification Report:
              precision    recall  f1-score   support
           0       0.90      0.80      0.85        45
           1       0.85      0.80      0.83        41
           2       0.71      0.71      0.71        48
           3       0.77      0.77      0.77        26
           4       0.75      0.52      0.61        29
           5       0.73      0.96      0.83        71
           6       0.97      0.89      0.93        35
           7       0.90      0.64      0.75        14
           8       0.73      0.39      0.51        28
           9       0.76      0.46      0.57        35
          10       1.00      0.93      0.97        15
          11       0.98      0.99      0.98        92
          12       0.93      0.79      0.86        34
          13       0.57      0.67      0.62         6
          14       0.85      0.91      0.88        43
          15       0.85      0.82      0.84        28
          16       0.88      0.88      0.88        51
          17       0.75      0.88      0.81       174

    accuracy                           0.82       815
    macro avg      0.83      0.77      0.79       815
    weighted avg   0.83      0.82      0.82       815

Execute Random Forest with best parameters: **83.56%**

# Run 3
Execution Time: 273s

UsedParameters: **[{'n_estimators': [10, 100, 500], 'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2']}]**

Best params: **{'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 500}**

Best score: **0.8167626238791883**

#### Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.87      0.90        45
           1       0.85      0.83      0.84        41
           2       0.77      0.75      0.76        48
           3       0.88      0.88      0.88        26
           4       0.89      0.55      0.68        29
           5       0.77      0.97      0.86        71
           6       1.00      0.89      0.94        35
           7       0.89      0.57      0.70        14
           8       0.71      0.43      0.53        28
           9       0.81      0.49      0.61        35
          10       1.00      0.87      0.93        15
          11       0.97      0.99      0.98        92
          12       0.90      0.76      0.83        34
          13       0.62      0.83      0.71         6
          14       0.85      0.93      0.89        43
          15       0.88      0.79      0.83        28
          16       0.87      0.88      0.87        51
          17       0.75      0.89      0.81       174

    accuracy                           0.84       815
    macro avg      0.85      0.79      0.81       815
    weighted avg   0.84      0.84      0.83       815


Execute Random Forest with best parameters: **83.56%**

# Run 4
Execution Time: 1094s

UsedParameters: **[{'n_estimators': [500, 1000, 2000], 'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2']}]**

Best params: **{'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 500}**

Best score: **0.8186078338839075**

#### Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.84      0.87        45
           1       0.85      0.80      0.83        41
           2       0.74      0.73      0.74        48
           3       0.85      0.88      0.87        26
           4       0.84      0.55      0.67        29
           5       0.76      0.96      0.85        71
           6       0.97      0.91      0.94        35
           7       0.90      0.64      0.75        14
           8       0.68      0.54      0.60        28
           9       0.78      0.40      0.53        35
          10       1.00      0.87      0.93        15
          11       0.98      0.99      0.98        92
          12       0.93      0.79      0.86        34
          13       0.67      0.67      0.67         6
          14       0.85      0.93      0.89        43
          15       0.88      0.82      0.85        28
          16       0.88      0.90      0.89        51
          17       0.76      0.89      0.82       174

    accuracy                           0.84       815
    macro avg      0.85      0.78      0.81       815
    weighted avg   0.84      0.84      0.83       815


Execute Random Forest with best parameters: **83.80%**
