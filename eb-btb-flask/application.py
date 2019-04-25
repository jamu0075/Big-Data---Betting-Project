#!/usr/bin/env python

from flask import Flask, render_template
import numpy as np
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# EB looks for an 'app' callable by default.
app = Flask(__name__)

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
        except:
            print('An error occured while attempting to connect to the database.')

        #print('Connected!')

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
        
# Home page
@app.route("/")
@app.route("/home")
def home():

    myDB = Database()
    leagues = myDB.get_leagues()
    teams = myDB.get_teams()

    return render_template('home.html', teams=teams, leagues=leagues, name="new_plot")


# About page
@app.route("/about")
def about():
    return render_template('about.html', title="About")

# Erik's test page
@app.route("/erikTest")
def erikTest():
    return render_template('erikTest.html', title="Erik Test")

@app.route('/plot.png')
def generate_plot():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig
    
# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
