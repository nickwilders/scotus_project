# These functions were created by the research team behind ________

# CITE TEAM


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, f1_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



def graph_by_CJ(justice):
    new_court = supreme[supreme['chief'] == justice]
    
    votes_issue = new_court.groupby('issueArea').decisionDirection.value_counts()
    
    sc_list = []

    for i, vote in enumerate(votes_issue.index):
        if i < 22:
            sc_list.append(votes_issue.index[i][0])

    vote_list = []

    for i, vote in enumerate(votes_issue.index):
        if i < 22:
            if votes_issue.index[i][1] == 0:
                    vote_list.append('Conservative')
            elif votes_issue.index[i][1] == 1:
                    vote_list.append('Liberal')

    count_list = []

    for i, vote in enumerate(votes_issue.index):
        if i < 22:
            count_list.append(votes_issue.iloc[i])
    
    if justice=='Warren':
        percent_count_list = []
        for i, sc in enumerate(count_list):
            if i < 8:
                if i % 2 == 0:
                    number = count_list[i] / (count_list[i] + count_list[i+1])
                    percent_count_list.append(number)
                else:
                    number = count_list[i] / (count_list[i] + count_list[i-1])
                    percent_count_list.append(number)
            elif i == 8:
                percent_count_list.append(1)
            elif i > 8:
                if i % 2 == 0:
                    number = count_list[i] / (count_list[i] + count_list[i-1])
                    percent_count_list.append(number)
                else:
                    number = count_list[i] / (count_list[i] + count_list[i+1])
                    percent_count_list.append(number)        

        
        percent_count_list = [n*100 for n in percent_count_list]
    else:
        percent_count_list = []
        for i, sc in enumerate(count_list):
            if i % 2 == 0:
                number = count_list[i] / (count_list[i] + count_list[i+1])
                percent_count_list.append(number)
            else:
                number = count_list[i] / (count_list[i] + count_list[i-1])
                percent_count_list.append(number)

        percent_count_list = [n*100 for n in percent_count_list]

            
    issues = ['Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process', 'Privacy', 'Attorneys', 'Unions',
         'Economic Activity', 'Judicial Power', 'Federalism', 'Federal Taxation']
    
    #%matplotlib inline
    import seaborn as sns

    plt.figure(figsize=(16,10))
    plt.title('')
    plt.ylim(0,100)
    ax = sns.barplot(x=sc_list, y=percent_count_list, hue=vote_list, palette='deep')
    plt.legend(prop={"size":15})
    ax.set_title('Supreme Court Issue Bias by Issue in Rehnquist Court (1986 - 2004)', fontsize=16, style='oblique')
    ax.set_xlabel('Issue', fontsize=14)
    ax.set_ylabel('% of Votes', fontsize=14, rotation = 0, horizontalalignment='right');
    ax.axhline(y=50, linewidth=3, color='red', alpha=.7, linestyle='--')
    ax.set_xticklabels(['Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process', 'Privacy', 'Attorneys', 'Unions',
             'Economic Activity', 'Judicial Power', 'Federalism', 'Federal Taxation'], rotation=45, fontsize=12)

    plt.savefig('Bias_by_Issue_in_Rehnquist_Court')



def find_scores(X, y, name, model_code):
    
    #Initialize test-train-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, 
                                         random_state=10)
    
    # set model variable
    model = model_code
    
    # fit individual model and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Accuracy and F1 Score
    score = metrics.accuracy_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, average='weighted')

    # Return Accuray, Precision and F1 score
    print("{name} accuracy is".format(name=name), score)
    print("Precision: {:6.4f},   Recall: {:6.4f}".format(precision_score(y_test, y_pred, average='weighted'), 
                                                         recall_score(y_test, y_pred, average='weighted')))
    print('f1 score: ', f1score)
    score = round(score,4)
    
    # Create Confusion Matrix
    plot_confusion_matrix(model, X_test, y_test)
    plt.title('Confusion Matrix for {name} - (Acc - {score})'.format(name=name, score=score));

















SCDB_OUTCOME_MAP=None

def get_outcome_map():
    
    import pandas as pd
    
    """
    Get the outcome map to convert an SCDB outcome into
    an affirm/reverse/other mapping.
    
    Rows correspond to vote types.  Columns correspond to disposition types.
    Element values correspond to:
    * -1: no precedential issued opinion or uncodable, i.e., DIGs
    * 0: affirm, i.e., no change in precedent
    * 1: reverse, i.e., change in precent
    """

    # Create map; see appendix of paper for further documentation
    outcome_map = pd.DataFrame([[-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
               [-1, 1, 0, 0, 0, 1, 0, -1, -1, -1, -1],
               [-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
               [-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
               [-1, 0, 1, 1, 1, 0, 1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
               [-1, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1]])
    outcome_map.columns = range(1, 12)
    outcome_map.index = range(1, 9)

    return outcome_map


def get_outcome(vote, disposition, outcome_map=SCDB_OUTCOME_MAP):
    
    import pandas as pd
    
    """
    Return the outcome code based on outcome map.
    """
    
    if not outcome_map:
        SCDB_OUTCOME_MAP=get_outcome_map()
        outcome_map = SCDB_OUTCOME_MAP

    if pd.isnull(vote) or pd.isnull(disposition):
        return -1
    
    return outcome_map.loc[int(vote), int(disposition)]