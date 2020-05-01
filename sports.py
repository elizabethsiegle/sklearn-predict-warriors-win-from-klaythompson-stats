from sklearn.metrics import confusion_matrix, classification_report,  accuracy_score
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request
csv = 'Seasons_Stats.csv'
cols = ['FG', 'FGA', 'FG%', '3P', '3PA', '3P%',
        'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB']
def setup_df(file, msg):
    df = pd.read_csv(csv)
    #print('df.columns ', df.columns)
    print(df.shape) # (83, 31) = 83 records/rows, 31 attributes/columns
    df = df[['H-A', 'FG%', '3P%', 'PTS', msg, 'WL']] #last item in arr is classifier

    #drop row with missing values
    df = df.fillna(0)

    df['WL'] = df['WL'].map({'W': 1, 'L': 0})
    df['H-A'] = df['H-A'].map({'H': 1, 'A': 0})

    # rmrow = df.loc[51:51, 'WLD':'+/-']  # startrow:endrow, startcol:endcol
    #0 axis = rows, 1 axis = columns
    m1 = df.eq('Did Not Dress').any(axis=1)
    m2 = df.eq('Did Not Dress').any(axis=0)
    m3 = df.eq('Inactive').any(axis=1)
    m4 = df.eq('Inactive').any(axis=0)
    #  NaN values might still have significance in being missing and imputing them with zeros is probably the worst thing you can do and the worst imputation method you use. Not only will you be introducing zeros arbitrarily which might skew your variable but 0 might not even be an acceptable value in your variables, meaning your variable might not have a true zero.
    df.loc[m1] = 0
    df.loc[:, m2] = 0
    df.loc[m3] = 0
    df.loc[:, m4] = 0
    return df

#  NaN values might still have significance in being missing and imputing them with zeros is probably the worst thing you can do and the worst imputation method you use. Not only will you be introducing zeros arbitrarily which might skew your variable but 0 might not even be an acceptable value in your variables, meaning your variable might not have a true zero.
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

app = Flask(__name__)
@app.route("/sms", methods=['GET', 'POST'])
def sms():
    resp = MessagingResponse()
    inb_msg = request.form['Body']
    if inb_msg in cols:
        df = setup_df(csv, inb_msg)
        clean_dataset(df)
        print(df.head())  # first 5 records of dataset  
        print(df)  # only data we're looking at with user input (also H-A, FG)
        X = df.drop('WL', axis=1)
        y = df['WL']
        # specifies ratio of test set, used to split up 20% of the data into test set and 80% for training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20)  # random_state=1
        model = tree.DecisionTreeClassifier()
        # print(model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print('accuracy score ', accuracy_score(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        scores = cross_val_score(model, X_test, y_test, scoring='accuracy')
        print('scores ', scores)
        print("Using just the last result from the home and visitor teams")
        acc_from_cross_val_score = np.mean(scores) * 100
        print("Accuracy: {}%".format(acc_from_cross_val_score))

        print('confusion matrix {}'.format(pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            columns=['Predicted Loss', 'Predicted Win'],
            index=['True Loss', 'True Win']
        )))
        # distributions for each group
        print('y_train class distribution')
        print(y_train.value_counts(normalize=True))
        print('y_test class distribution')
        print(y_test.value_counts(normalize=True))
        msg = 'accuracy score: {}\n confusion matrix: {}\n distributions for predicted group: {}\n'.format(
            acc_from_cross_val_score, confusion_matrix(y_test, y_pred), y_test.value_counts(normalize=True))
        
    else:
        msg = "Send a message according to Klay Thompson's stats columns: {}".format(cols)
    resp.message(msg)
    return str(resp)
