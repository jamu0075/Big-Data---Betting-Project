#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
import os
import pymysql
from flaskext.mysql import MySQL
import numpy as np
import pandas as pd
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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
        sql = 'SELECT DISTINCT league FROM outcomeFeatures ORDER BY league'
        self.curs.execute(sql)
        result = self.curs.fetchall()

        return result

    # Return unique teams from the db
    def get_teams(self):
        sql = 'SELECT DISTINCT home FROM outcomeFeatures ORDER BY home'
        self.curs.execute(sql)
        result = self.curs.fetchall()

        return result

      # Return unique teams in a given league from the db
    def get_teams_in_league(self, league):
        sql = "SELECT DISTINCT home FROM outcomeFeatures WHERE league = '{}' ORDER BY home".format(league) # .format puts league into {}
        self.curs.execute(sql)
        result = self.curs.fetchall()

        return result

    def get_teams_record(self, team1, team2):
        teams = (team1, team2)
        sql = "SELECT home, away, winner, home_closing, away_closing FROM outcomeFeatures WHERE home IN {} AND away IN {}".format(teams, teams)

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


# Nick create this
@app.route("/nickdev")
def nickdev():

    myForm = Form()
    # Fetch form values from the database
    myForm.league.choices = [(league[0], league[0]) for league in myDB.get_leagues()]
    myForm.team1.choices = [(team[0], team[0]) for team in myDB.get_teams()]
    myForm.team2.choices = [(team[0], team[0]) for team in myDB.get_teams()]

    if request.method == 'POST':
        print('FORM RECIEVED')
        return '<h1>League: {}, Team1: {}, Team2: {}</h1>'.format(form.league.data, form.team1.data, form.team2.data)

    return render_template('nickdev.html', form=myForm)


@app.route('/plot.png')
def plot_png():
    fig = create_team_records_fig()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_team_records_fig():
    df = myDB.get_teams_records()
    # update these:
    team1 = 'Everton'
    n_games = df['home'].size
    # x is integers from 1 to n_games for home teams then 1 to n_games for away
    x = np.concatenate([np.linspace(1, n_games, n_games)] * 2)
    # y is home_teams first at y=1 then away teams at y=y_height
    y_height = 3
    y = np.concatenate([np.ones(n_games), y_height * np.ones(n_games)])
    # sizes of bubbles are odds of each team
    sizes = np.concatenate([df['home_closing'], df['away_closing']])
    # names of teams
    team_names = np.concatenate([df['home'], df['away']])
    # colors is the color of each bubble where team1 = blue and team2  = red
    colors = ['grey'] * n_games * 2
    i = 0  # i is the row of df
    while i < n_games:
        outcome = df['winner'].values[i]
        if outcome == 'home':
            team = df['home'].values[i]
            if team == team1:
                colors[i] = 'blue'
            else:
                colors[i] = 'red'
        elif outcome == 'away':
            team = df['away'].values[i]
            if team == team1:
                colors[n_games + i] = 'blue'
            else:
                colors[n_games + i] = 'red'
        i = i + 1


    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.scatter(x, y, s=sizes * 1000 * (6 - n_games), c=colors, alpha=0.4)
    axis.set_xlim([0, n_games + 1])
    axis.set_ylim([0, y_height + 1])
    axis.set_xticks(x[0:n_games])
    # update this to be dates
    axis.set_xticklabels(['99-99-99', '99-99-99'])
    axis.set_yticks([1, y_height])
    axis.set_yticklabels(['home', 'away'])
    i = 0
    while i < n_games * 2:
        axis.text(x[i], y[i], team_names[i],
                     horizontalalignment='center',
                     verticalalignment='center')
        i = i + 1
    return fig



# run the app.
if __name__ == "__main__":

    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
