#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify, Response
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
import os
import pymysql
from flaskext.mysql import MySQL

# Nick added these
import numpy as np
import pandas as pd
import io
import matplotlib
matplotlib.use('agg')
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
        user = 
        password =
        db =

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
        sql = "SELECT match_date, home_team, away_team, winning_team, home_closing, away_closing FROM outcomeFeatures2 WHERE home_team IN {} AND away_team IN {}".format(teams, teams)
        self.curs.execute(sql)
        result = self.curs.fetchall()
        return result

    # Nick added this to set global variabels
    def set_full_team_records(self, team1, team2):
        sql1 = "SELECT winning_team, closing_odds_outcome FROM outcomeFeatures2 WHERE home_team = '{}' OR away_team = '{}' ORDER BY match_date".format(team1, team1)
        self.curs.execute(sql1)
        teams.team1_full_record = self.curs.fetchall()
        sql2 = "SELECT winning_team, closing_odds_outcome FROM outcomeFeatures2 WHERE home_team = '{}' OR away_team = '{}' ORDER BY match_date".format(team2, team2)
        self.curs.execute(sql2)
        teams.team2_full_record = self.curs.fetchall()

# A dynamic Form object for UI
# This is where Jacob gets the stuff:
class Form(FlaskForm):
    league = SelectField('League', choices=[])
    team1 = SelectField('Team 1', choices=[])
    team2 = SelectField('Team 2', choices=[])
    submit = SubmitField('Submit')

# global class that nick created for visuals
class Teams:
    def __init__(self):
        team_records = []
        team1_full_record = []
        team2_full_record = []
        team1 = ''
        team2 = ''

#=========================================================
# A database connection instance for global use
myDB = Database()
teams = Teams() # Nick needs this for his visuals
#=========================================================

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():


    myForm = Form(request.form)
    #Fetch form values from the database
    myForm.league.choices = [(league[0], league[0]) for league in myDB.get_leagues()]
    myForm.team1.choices = [(team[0], team[0]) for team in myDB.get_teams()]
    myForm.team2.choices = [(team[0], team[0]) for team in myDB.get_teams()]

    # After you presss submit,  Handle form POST, update page
    # league team1 and team 2 coming from the form POST
    # then passes to class='container graph" section in home html
    if request.method == 'POST':
        teams_record = myDB.get_teams_record(myForm.team1.data, myForm.team2.data)
        # next four lines set the global variables that nick uses for visuals
        teams.team1 = myForm.team1.data
        teams.team2 = myForm.team2.data
        teams.team_records = teams_record
        myDB.set_full_team_records(teams.team1,  teams.team2)
        return render_template('home.html', form=myForm, league=myForm.league.data, team1=myForm.team1.data, team2=myForm.team2.data,
                               teams_record=teams_record)

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
# @app.route("/about")
# def about():
#     return render_template('about.html', title="About")
#

# @app.route("/nickdev")
# def nickdev():
#     return render_template('nickdev.html', plot_title = (teams.team1 + " vs. " + teams.team2))


@app.route('/team_records')
def plot_team_records():
    # fig is created in function below
    fig = create_team_records_fig(teams.team_records, teams.team1, teams.team2)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def create_team_records_fig(team_records, team1, team2):
    df = pd.DataFrame(data=list(team_records), columns=['match_date', 'home_team', 'away_team', 'winning_team', 'home_closing', 'away_closing']) # need list of tuples instead of tuple of tuples
    print(df)
    #print('1 got past')
    n_games = df['home_team'].size
    # x is integers from 1 to n_games for team1 then 1 to n_games for team2
    x = np.concatenate([np.linspace(1, n_games, n_games)] * 2)
    # y is home_teams first at y=1 then away teams at y=y_height
    y_height = 3
    y = np.concatenate([np.ones(n_games), y_height * np.ones(n_games)])
    # sizes of bubbles are odds of each team
    sizes = np.concatenate([df['home_closing'], df['away_closing']])
    # colors is the color of each bubble where team1 = blue and team2  = red if they win
    colors = ['grey']*n_games*2
    bubble_text = [''] * n_games * 2
    i = 0
    while i < n_games: # i is the row of df
        if df['home_team'].values[i] == team1:
            bubble_text[i] = str(df['home_closing'].values[i]) + '\n home'
            sizes[i] = df['home_closing'].values[i]
            bubble_text[i+n_games] = str(df['away_closing'].values[i]) + '\n away'
            sizes[i+n_games] = df['away_closing'].values[i]
        else:
            bubble_text[i] = str(df['away_closing'].values[i]) + '\n away'
            sizes[i] = df['away_closing'].values[i]
            bubble_text[i+n_games] = str(df['home_closing'].values[i]) + '\n home'
            sizes[i+n_games] = df['home_closing'].values[i]
        winner = df['winning_team'].values[i]
        if winner == team1:
            colors[i] = 'green'
        elif winner == team2:
            colors[i+n_games] = 'green'
        i = i+1
        print(i)

    fig = plt.figure(figsize=(8, 12), tight_layout = True)
    ax1 = fig.add_subplot(2,1,1)
    ax1.scatter(x, y, s=sizes * 1000 * (6 - n_games), c=colors, alpha=0.4)
    ax1.set_xlim([0, n_games + 1])
    ax1.set_ylim([0, y_height + 1])
    ax1.set_xticks(x[0:n_games])
    ax1.set_xticklabels(df['match_date'])
    ax1.set_yticks([1, y_height])
    ax1.set_yticklabels([team1, team2])
    ax1.set_title('Final Odds with Outcome')

    i = 0
    while i < n_games * 2:
        ax1.text(x[i], y[i], bubble_text[i],
                     horizontalalignment='center',
                     verticalalignment='center')
        i = i + 1

    # Next plot
    df = pd.DataFrame(data=list(teams.team1_full_record), columns = ['winning_team', 'closing_odds_outcome'])
    n_games = len(df['winning_team'])
    size_of_each_bet = 100
    return_on_bet = (df['winning_team'] == team1)*df['closing_odds_outcome']*size_of_each_bet
    winnings = np.zeros(n_games)
    winnings[0] = -size_of_each_bet + return_on_bet.values[0]
    i = 1
    while i < n_games:
        winnings[i] = winnings[i-1] - size_of_each_bet + return_on_bet.values[i]
        i = i+1
    x = np.linspace(1, n_games, n_games)
    fig.add_subplot(2,1,2)
    plt.plot(x,winnings, c = 'grey')
    plt.hlines(y=0, xmin=0, xmax = n_games)
    green = winnings > 0
    colors = ['']*len(green)
    for i in range(len(green)):
        if green[i] == True:
            colors[i] = 'g'
        else:
            colors[i] = 'r'
    plt.scatter(x, winnings, c = colors)

    # last scatter plot on same axes as first scatter
    df = pd.DataFrame(data=list(teams.team2_full_record), columns = ['winning_team', 'closing_odds_outcome'])
    n_games = len(df['winning_team'])
    size_of_each_bet = 100
    return_on_bet = (df['winning_team'] == team2)*df['closing_odds_outcome']*size_of_each_bet
    winnings = np.zeros(n_games)
    winnings[0] = -size_of_each_bet + return_on_bet.values[0]
    i = 1
    while i < n_games:
        winnings[i] = winnings[i-1] - size_of_each_bet + return_on_bet.values[i]
        i = i+1
    x = np.linspace(1, n_games, n_games)

    green = winnings > 0
    colors = ['']*len(green)
    for i in range(len(green)):
        if green[i] == True:
            colors[i] = 'g'
        else:
            colors[i] = 'r'

    fig.add_subplot(2,1,2)
    plt.plot(x,winnings, c = 'grey', linestyle = 'dashed')
    plt.hlines(y=0, xmin=0, xmax = n_games)
    plt.scatter(x, winnings, c = colors, marker = 'v')
    plt.legend(labels = [teams.team1, teams.team2])
    plt.title('Return when betting $100 each game', fontsize=16)
    plt.xlabel('Games')
    plt.ylabel('Total Winnings')
    plt.close() # plt.close() is needed or my mac gets a weird error
    return fig



# run the app.
if __name__ == "__main__":

    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
    print('got to running the app')
