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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
######################################

# EB looks for an 'app' callable by default.
app = Flask(__name__)

#wtf secret key
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

# A database class to use the DB as an object
class Database:
    def __init__(self):
        host = 'btb-db-instance.cduiw3cccdch.us-east-1.rds.amazonaws.com'
        port = 3306
        user = 'masterUser'
        password = 'supremedbpass2002'
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
    # sql = "SHOW columns FROM outcomeFeatures2"
    # myDB.curs.execute(sql)
    # print(myDB.curs.fetchall())
    myForm = Form(request.form)
    #Fetch form values from the database
    myForm.league.choices = [(league[0], league[0]) for league in myDB.get_leagues()]
    # print(myForm.league.choices)
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
@app.route('/erikTest', methods=['GET', 'POST'])
def index():
    leagues = [{'name': league[0]} for league in myDB.get_leagues()]
    return render_template(
        'submit.html',
        data=leagues, title="Erik Test")

@app.route("/response" , methods=['GET', 'POST'])
def response():
    sql = "SHOW columns FROM outcomeFeatures2"
    myDB.curs.execute(sql)
    print(myDB.curs.fetchall())

    league_selected = request.form.get('comp_select')
    columns_wanted = ", ".join(['outcome', 'home_opening', 'home_min'])
    sql = "SELECT {} FROM outcomeFeatures2 WHERE league = '{}'".format(columns_wanted, league_selected)
    # myDB.curs.execute(sql)
    # print(myDB.curs.fetchall())
    df = pd.read_sql(sql, myDB.conn)
    print(df)
    model, X_test = construct_ml_model(df)
    test_data = pd.DataFrame({'home_opening': [3.01], 'home_min': [1.01]})
    print(test_data)
    print(model.predict(test_data))
    # print(X_test)
    return render_template('response.html', title=league_selected) # just to see what select is

@app.route('/plot.txt')
def generate_plot():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [np.random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig

def construct_ml_model(df):
    # Set hyper parameter values
    num_trees = 10 # number of trees in random forest
    test_prop = 0.1 # proportion of data to use for testing
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
    rf_clf = RandomForestClassifier(n_estimators=num_trees)
    rf_clf.fit(X_train, y_train)
    return(rf_clf, X_test)

######################################################################
######################################################################

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
