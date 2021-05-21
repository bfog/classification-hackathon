# Classification Hackathon
A 2 week hackathon conducted to develop a classification learning agent which is able to classify unknown platforms

## GridSearchCV results

### Configuration:
* Route used: Route 1
* Window size: 300
* Step size: 100
* Test split: 0.2
* Use unknowns: Yes


### Base scores:
* Gaussian NB: 11.78%
* SVM: 39.51%
* Decision Tree: 70.92%
* NCA & KNN: 40.74%
* Random Forest: 81.96%

GridSearchCV was used for Random Forest only, as it has the highest base scoring. The aim is to improve this score by finding the optimal parameters.
Random Forest base parameters were set to the default values: n_estimators: 100, random_state: 1

# Run 1

# Run 2

# Run 3
Execution Time: 159.33709740638733

UsedParameters [{'n_estimators': [140, 150, 160], 'criterion': ['gini', 'entropy'], 'max_features': ['sqrt', 'log2']}]
Best params:
{'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 160}
Best score: 0.8146210476639923
#### Classification Report:
       |        | precision |   recall | f1-score  | support |
       |        | --------  | -------- | --------  | ------- |
       |    0   |    0.90   |   0.80   |   0.85    |    45   |
       |    1   |    0.85   |   0.80   |   0.83    |    41   |
       |    2   |    0.71   |   0.71   |   0.71    |    48   |
       |    3   |    0.77   |   0.77   |   0.77    |    26   |
       |    4   |    0.75   |   0.52   |   0.61    |    29   |
       |    5   |    0.73   |   0.96   |   0.83    |    71   |
       |    6   |    0.97   |   0.89   |   0.93    |    35   |
       |    7   |    0.90   |   0.64   |   0.75    |    14   |
       |    8   |    0.73   |   0.39   |   0.51    |    28   |
       |    9   |    0.76   |   0.46   |   0.57    |    35   |
       |   10   |    1.00   |   0.93   |   0.97    |    15   |
       |   11   |    0.98   |   0.99   |   0.98    |    92   |
       |   12   |    0.93   |   0.79   |   0.86    |    34   |
       |   13   |    0.57   |   0.67   |   0.62    |     6   |
       |   14   |    0.85   |   0.91   |   0.88    |    43   |
       |   15   |    0.85   |   0.82   |   0.84    |    28   |
       |   16   |    0.88   |   0.88   |   0.88    |    51   |
       |   17   |    0.75   |   0.88   |   0.81    |   174   |
                |           |          |           |         |
    accuracy    |           |          |   0.82    |   815   |
   macro avg    |    0.83   |   0.77   |   0.79    |   815   |
weighted avg    |    0.83   |   0.82   |   0.82    |   815   |

Execute Random Forest with best parameters: 83.31%
