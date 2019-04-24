from flask import Flask, render_template
import os
import pymysql
from flaskext.mysql import MySQL

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

    # Nick create this
    def display_plot(self):
        return
        

# Home page
@app.route("/")
@app.route("/home")
def home():

    myDB = Database()
    leagues = myDB.get_leagues()
    teams = myDB.get_teams()

    return render_template('home.html', teams=teams, leagues=leagues) # pass in arguments teams and leagues


# About page
@app.route("/about")
def about():
    return render_template('about.html', title="About")

# Nick create this
@app.route("/nickdev")
def nickdev():

    myDB = Database()
    leagues = myDB.get_leagues()
    teams = myDB.get_teams()
    
    # this is the last thing to do
    return render_template('nickdev.html', teams=teams, leagues=leagues)


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
