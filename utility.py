# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 20:09:20 2021

@author: mxkep
"""
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, make_scorer
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd

def print_table(y_test, y_pred_dt, y_pred_rf, y_pred_nb):
    
    
    final_table = PrettyTable()
    final_table.field_names = ['Model Name',
                  'Accuracy',
                  'F1-Score',
                  'Precision',
                  'Recall'
                 ]

    final_table.add_row([
    'Decision Tree',
    accuracy_score(y_true = y_test, y_pred = y_pred_dt),
    f1_score(y_true = y_test, y_pred = y_pred_dt),
    precision_score(y_true = y_test, y_pred = y_pred_dt),
    recall_score(y_true = y_test, y_pred = y_pred_dt)
    ])

    final_table.add_row([
    'Random Forest',
    accuracy_score(y_true = y_test, y_pred = y_pred_rf),
    f1_score(y_true = y_test, y_pred = y_pred_rf),
    precision_score(y_true = y_test, y_pred = y_pred_rf),
    recall_score(y_true = y_test, y_pred = y_pred_rf)
    ])

    final_table.add_row([
    'Naive Bayes',
    accuracy_score(y_true = y_test, y_pred = y_pred_nb),
    f1_score(y_true = y_test, y_pred = y_pred_nb),
    precision_score(y_true = y_test, y_pred = y_pred_nb),
    recall_score(y_true = y_test, y_pred = y_pred_nb)
    ])
 
    print(final_table)
    
    
def print_table_svm(y_test, y_pred_poly, y_pred_rbf, string1, string2):
    
    
    final_table = PrettyTable()
    final_table.field_names = ['Model Name',
                  'Accuracy'
                 ]

    final_table.add_row([
        string1,
        accuracy_score(y_true = y_test, y_pred = y_pred_poly)
        ])
    
    final_table.add_row([
        string2,
        accuracy_score(y_true = y_test, y_pred = y_pred_rbf)
        ])

    print(final_table)
    

def pretty_importances_plot(importances, feature_name, fig_size = (10, 7), xlabel = '', ylabel = '', horizontal_label = None, n_elements=None):
    '''
    This function plots a better looking importances-plot
    
    importances: Occurences of feature_name
    feature_name: unique feature_name
    fig_size: size of plot
    xlabel: xlabel
    ylabel: ylabel
    horizontal_label: Bigger label at the top-left
    n_elements: number of elements to display
    '''
    
    # This code has been borrowed from:
    # https://scentellegher.github.io/visualization/2018/10/10/beautiful-bar-plots-matplotlib.html
    # Credits to SIMONE CENTELLEGHER

    # set the style of the axes and the text color
    plt.rcParams['axes.edgecolor']='#333F4B'
    plt.rcParams['axes.linewidth']=0.8
    plt.rcParams['xtick.color']='#333F4B'
    plt.rcParams['ytick.color']='#333F4B'
    plt.rcParams['text.color']='#333F4B'

    # percentages = pd.Series(data= list(map(operator.itemgetter(1), word_importances_title_entity)),
    #                       index = list(map(operator.itemgetter(0), word_importances_title_entity)))
    
    
    if not n_elements:
        n_elements = len(importances)
    
    percentages = pd.Series(
        data = importances[:n_elements],
        index = feature_name[:n_elements]
    )

    df = pd.DataFrame({'percentage' : percentages})
    df = df.sort_values(by='percentage')

    # we first need a numeric placeholder for the y axis
    my_range=list(range(1,len(df.index)+1))

    #fig, ax = plt.subplots(figsize=(5,3.5)) ######################
    fig, ax = plt.subplots(figsize=fig_size)


    # create for each expense type an horizontal line that starts at x = 0 with the length 
    # represented by the specific expense percentage value.
    plt.hlines(y=my_range, xmin=0, xmax=df['percentage'], color='#007ACC', alpha=0.2, linewidth=5)

    # create for each expense type a dot at the level of the expense percentage value
    plt.plot(df['percentage'], my_range, "o", markersize=5, color='#007ACC', alpha=0.6)

    # set labels
    ax.set_xlabel(xlabel, fontsize=15, fontweight='black', color = '#333F4B')
    ax.set_ylabel(ylabel)

    # set axis
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.yticks(my_range, df.index)

    # add an horizonal label for the y axis
    if horizontal_label:
        fig.text(-0.23, 0.96, horizontal_label, fontsize=15, fontweight='black', color = '#333F4B')

    # change the style of the axis spines
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # set the spines position
    ax.spines['bottom'].set_position(('axes', -0.04))
    ax.spines['left'].set_position(('axes', 0.015))
    
    
def cv_model(train, train_labels, model, name, model_results=None):
    """
    @author: WILL KOEHRSEN - Kaggle - "A Complete Introduction and Walkthrough"
    REFerences:
      https://www.kaggle.com/willkoehrsen/a-complete-introduction-and-walkthrough#Model-Selection
    """
    
    """Perform 10 fold cross validation of a model"""
    scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')

    cv_scores = cross_val_score(model, train, train_labels, cv = 10, scoring=scorer, n_jobs = -1)
    print(f'10 Fold CV Score: {round(cv_scores.mean(), 5)} with std: {round(cv_scores.std(), 5)}')
    
    if model_results is not None:
        model_results = model_results.append(pd.DataFrame({'model': name, 
                                                           'cv_mean': cv_scores.mean(), 
                                                            'cv_std': cv_scores.std()},
                                                           index = [0]),
                                             ignore_index = True)

        return model_results