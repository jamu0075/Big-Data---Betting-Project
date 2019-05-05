#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
import os
import pymysql
from flaskext.mysql import MySQL

######################################
# Erik Added
import numpy as np
import io
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.utils.multiclass import unique_labels
from sklearn.externals import joblib

######################################

# EB looks for an 'app' callable by default.
app = Flask(__name__)

#wtf secret key
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

times_wanted = ['0', '23', '47', '71']
predictors_wanted = [['home_'+time, 'away_'+time, 'draw_'+time] for time in times_wanted]
predictors_wanted = [item for sublist in predictors_wanted for item in sublist]
colms_wanted = predictors_wanted.copy()
colms_wanted.append('outcome')
colms_wanted_as_str = ', '.join(colms_wanted)

# A database class to use the DB as an object
class Database:
    def __init__(self):
        host = 'btb-db-instance.cduiw3cccdch.us-east-1.rds.amazonaws.com'
        port = 3306
        user = 
        password = 
        db = 'btb-db'

        print('Attempting to connect...')
        try:
            self.conn = pymysql.connect(host, user=user, port=port, passwd=password, db=db)
            self.curs = self.conn.cursor()
            print('Connected!')
        except:
            print('An error occured while attempting to connect to the database.')


    # Return unique leagues from the db
    def get_leagues(self):
        sql = 'SELECT DISTINCT league FROM outcomeFeatures2 ORDER BY league'
        self.curs.execute(sql)
        result = self.curs.fetchall()

        return result

    # Return unique teams from the db
    def get_teams(self):
        sql = 'SELECT DISTINCT home_team FROM outcomeFeatures2 ORDER BY home_team'
        self.curs.execute(sql)
        result = self.curs.fetchall()

        return result

    # Return unique teams in a given league from the db
    def get_teams_in_league(self, league):
        sql = "SELECT DISTINCT home_team FROM outcomeFeatures2 WHERE league = '{}' ORDER BY home_team".format(league)
        self.curs.execute(sql)
        result = self.curs.fetchall()

        return result

    def get_teams_record(self, team1, team2):
        teams = (team1, team2)
        sql = "SELECT match_date, home_team, away_team, winning_team FROM outcomeFeatures2 WHERE home_team IN {} AND away_team IN {}".format(teams, teams)

        self.curs.execute(sql)
        result = self.curs.fetchall()

        return result

# A dynamic Form object for UI
class Form(FlaskForm):
    league = SelectField('League', choices=[])
    team1 = SelectField('Team 1', choices=[])
    team2 = SelectField('Team 2', choices=[])

    submit = SubmitField('Submit')

#=========================================================
# A database connection instance for global use
myDB = Database()
#=========================================================

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    myForm = Form(request.form)

    #Fetch form values from the database
    myForm.league.choices = [(league[0], league[0]) for league in myDB.get_leagues()]
    myForm.team1.choices = [(team[0], team[0]) for team in myDB.get_teams()]
    myForm.team2.choices = [(team[0], team[0]) for team in myDB.get_teams()]

    # Handle form POST, update page
    if request.method == 'POST':
        teams_record = myDB.get_teams_record(myForm.team1.data, myForm.team2.data)

        return render_template('home.html', form=myForm, league=myForm.league.data, team1=myForm.team1.data, team2=myForm.team2.data, teams_record=teams_record)

    return render_template('home.html', form=myForm)


#Route to handle dynamic dropdown
@app.route('/<league>')
def team(league):

    # Get teams from user inputed league
    teams = myDB.get_teams_in_league(league)

    # Create a list of dictionary objects for dropdown
    teamList = []

    for team in teams:
        teamObj = {}
        teamObj['name'] = team[0]
        teamList.append(teamObj)

    return jsonify({'teams' : teamList})

# About page
@app.route("/about")
def about():
    return render_template('about.html', title="About")


#########################################################
# Erik Added Code
#########################################################

# Erik's test page
@app.route('/betting', methods=['GET', 'POST'])
def betting_home():

    # Retrieve possible leagues from filtered data
    sql = 'SELECT DISTINCT league FROM filteredMatches ORDER BY league'
    myDB.curs.execute(sql)
    leagues = myDB.curs.fetchall()

    # Put data in format for passing to html
    leagues = [{'name': league[0]} for league in leagues]

    return render_template('submit.html', data=leagues, title="Betting")

@app.route("/response" , methods=['GET', 'POST'])
def response():

    # Select predictors and response columns from database
    league_selected = request.form.get('league_select')

    # Get wanted columns from filtered data
    sql = "SELECT {} FROM filteredMatches WHERE league = '{}'".format(colms_wanted_as_str, league_selected)
    df = pd.read_sql(sql, myDB.conn)
    df = df[colms_wanted] # make sure order of columns is consistent

    # Split the data into a training set and a test set
    # below keep random_state=0 for testing so get same thing everytime
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # classifier, conf_mat = construct_random_forest_model(df)
    construct_random_forest_model(df)

    return render_template('response.html', league=league_selected)


@app.route('/odds_form_on_submit/<string:league>', methods=['POST'])
def odds_form_on_submit(league):
# @app.route('/test_form_submit', methods=['POST'])
# def test_form_submit():
    num_folds = 20
    results_dict = request.form.to_dict()
    keys = list(results_dict.keys())
    values = list(results_dict.values())
    results_dict = dict((key, convert_str_to_float(value)) for key, value in results_dict.items())

    X_test = pd.DataFrame(results_dict, index=[0])
    X_test = X_test[predictors_wanted]

    classifier = joblib.load('model.pkl')
    prediction = classifier.predict(X_test)[0]

    sql = "SELECT {} FROM filteredMatches WHERE league = '{}'".format(colms_wanted_as_str, league)
    df = pd.read_sql(sql, myDB.conn)
    df = df[colms_wanted]

    X = df.drop('outcome', axis=1)
    y = df['outcome']


    y_low_odds = predict_lowest_closing_odds(X)
    acc_low_odds = accuracy_score(y, y_low_odds)
    acc_rf = cross_val_score(classifier, X, y, cv=num_folds).mean()

    f1_rf = cross_val_score(classifier, X, y, cv=num_folds, scoring='f1_weighted').mean()
    f1_low_odds = f1_score(y, y_low_odds, average='weighted')

    return render_template('partials/oddsFormResults.html', 
        keys=keys,
        prediction=prediction, 
        values=values, 
        league=league, 
        acc_low_odds=round(acc_low_odds, 2),
        acc_rf=round(acc_rf, 2),
        f1_rf=round(f1_rf, 2),
        f1_low_odds=round(f1_low_odds, 2)
        )

@app.route('/conf_mat.txt/<string:league>')
def generate_conf_mat(league):
    sql = "SELECT {} FROM filteredMatches WHERE league = '{}'".format(colms_wanted_as_str, league)
    df = pd.read_sql(sql, myDB.conn)
    df = df[colms_wanted]
    fig = create_conf_mat(league, df)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


###########################################################
# HELPER FUNCTIONS
def convert_str_to_float(value):
    try:
        return float(value)
    except:
        return 0.0

def create_conf_mat(league, df, num_folds=20):
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    class_names = unique_labels(y)

    classifier = joblib.load('model.pkl')

    y_pred = cross_val_predict(classifier, X, y, cv=num_folds)

    fig = plot_confusion_matrix(y, y_pred, classes=class_names, normalize=False)

    return fig

def construct_random_forest_model(df):
    # Set hyper parameter values
    num_trees = 20 # number of trees in random forest
    test_prop = 0.1 # proportion of data to use for testing
    num_folds = 10 # number of cross-validation folds

    # Construct X and y matrices for ML
    X = df.drop('outcome', axis=1)
    y = df['outcome']

    rf_clf = RandomForestClassifier(n_estimators=num_trees)

    # y_pred = cross_val_predict(rf_clf, X, y, cv=num_folds)

    rf_clf.fit(X, y)

    # Save model
    joblib.dump(rf_clf, 'model.pkl')

    return None

def predict_lowest_closing_odds(X):
    label_dict = {0: 'home', 1: 'away', 2: 'draw'}
    predictions = X.apply(lambda x: np.argmin([x['home_0'], x['away_0'], x['draw_0']]), axis=1)
    predictions = predictions.apply(lambda x: label_dict[x])
    return(np.array(predictions))

# def plot_confusion_matrix(y_true, y_pred, classes, accuracy,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized Confusion Matrix'
        else:
            title = 'Confusion Matrix without Normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")

    # accuracy = sum(y_pred==y_true)/len(y_true)

    fig = Figure()
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           # title=title + "  (N={}, Accuracy={})".format(len(y_pred), round(accuracy,2)),
           title=title + "  (Total Num Games={})".format(len(y_pred)),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

######################################################################
######################################################################

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
