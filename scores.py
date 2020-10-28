def find_scores(name, model_code):
    
    model = model_code
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Use OneHotEncoder to achieve binary variables for Democrat and Republican

    from numpy import asarray
    from sklearn.preprocessing import OneHotEncoder

    y_test1 = asarray(y_test).reshape(-1,1)
    y_pred1 = asarray(y_pred).reshape(-1,1)

    encoder = OneHotEncoder(sparse=False)
    y_test_encoded = encoder.fit_transform(y_test1)
    y_pred_encoded = encoder.fit_transform(y_pred1)
    print(y_pred_encoded.shape)
    
    # find scores

    score = metrics.accuracy_score(y_test_encoded, y_pred_encoded)
    f1score = f1_score(y_test_encoded, y_pred_encoded, average='weighted')

    print("{name} accuracy is".format(name=name), score)
    print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_test_encoded, y_pred_encoded, average='weighted'), 
                                                         recall_score(y_test_encoded, y_pred_encoded, average='weighted')))
    print('f1 score: ', f1score)
    
    clf = model_code
    clf.fit(X_train, y_train)
    plot_confusion_matrix(clf, X_test, y_test);
    
    
    
def all_scores(): 
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    %matplotlib inline

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score, plot_confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    
    def find_scores(name, model_code):
    
    model = model_code
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Use OneHotEncoder to achieve binary variables for Democrat and Republican

    from numpy import asarray
    from sklearn.preprocessing import OneHotEncoder

    y_test1 = asarray(y_test).reshape(-1,1)
    y_pred1 = asarray(y_pred).reshape(-1,1)

    encoder = OneHotEncoder(sparse=False)
    y_test_encoded = encoder.fit_transform(y_test1)
    y_pred_encoded = encoder.fit_transform(y_pred1)
    print(y_pred_encoded.shape)
    
    # find scores

    score = metrics.accuracy_score(y_test_encoded, y_pred_encoded)
    f1score = f1_score(y_test_encoded, y_pred_encoded, average='weighted')

    print("{name} accuracy is".format(name=name), score)
    print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_test_encoded, y_pred_encoded, average='weighted'), 
                                                         recall_score(y_test_encoded, y_pred_encoded, average='weighted')))
    print('f1 score: ', f1score)
    
    clf = model_code
    clf.fit(X_train, y_train)
    plot_confusion_matrix(clf, X_test, y_test);
    
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    algs = ['LogReg', 'KNN', 'Gaussian', 'SVC', 'Decision Tree', 'Random Forest']
    
    classifiers = [LogisticRegression(solver='lbfgs'), 
               KNeighborsClassifier(n_neighbors=10),
               GaussianNB(), 
               SVC(),
               DecisionTreeClassifier(),
               RandomForestClassifier()]
    for cls in classifiers:
        cls.fit(X_train, y_train)
        
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

    for cls, ax in zip(classifiers, axes.flatten()):
        plot_confusion_matrix(cls, 
                              X_test, 
                              y_test, 
                              ax=ax, 
                              cmap='Blues',
                             display_labels=algs)
        ax.title.set_text(type(cls).__name__)
    plt.tight_layout()  
    plt.show()

