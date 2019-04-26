from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from wtforms import SelectField
import os
import pymysql
from flaskext.mysql import MySQL
import numpy as np
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

    # get whole record of team1 vs team2
    def get_teams_records(self):
        tms = ('Everton', 'Liverpool')
        sql = "SELECT home, away, winner FROM outcomeFeatures WHERE home IN {} AND away IN {}".format(tms, tms)
        self.curs.execute(sql)
        result = self.curs.fetchall()
        print(result)

    # Nick create this
    #def display_plot(self):
    #    return
        

# A dynamic Form object for UI
class Form(FlaskForm):
    league = SelectField('league', choices=[])
    team1 = SelectField('team1', choices=[])
    team2 = SelectField('team2', choices=[])


# A database connection instance for global use
myDB = Database()

# Home page
@app.route("/", methods=['GET', 'POST'])
@app.route("/home")
def home():

    myForm = Form()
    # Fetch form values from the database
    myForm.league.choices = [(league[0], league[0]) for league in myDB.get_leagues()]
    myForm.team1.choices = [(team[0], team[0]) for team in myDB.get_teams()]
    myForm.team2.choices = [(team[0], team[0]) for team in myDB.get_teams()]

    if request.method == 'POST':
        print('FORM RECIEVED')
        return '<h1>League: {}, Team1: {}, Team2: {}</h1>'.format(form.league.data, form.team1.data, form.team2.data)

    return render_template('home.html', form=myForm)


#Route to handle dynamic dropdown
@app.route('/team/<league>')
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
    myDB.get_teams_records()

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
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = plt.figure()
    #fig = Figure()
    n_games = 100
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys, 'r')
    axis.set_xlim([0, n_games+1])
    return fig


# run the app.
if __name__ == "__main__":

    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
