from flask import Flask, render_template
import os
import pymysql
from flaskext.mysql import MySQL

# EB looks for an 'app' callable by default.
app = Flask(__name__)

#Temp data to emulate database response data
teams = [
    {
        'name': 'Team 1',
        'other': 'Team 1 info'
    },
    {
        'name': 'Team 2',
        'other': 'Team 2 info'
    },
    {
        'name': 'Team 3',
        'other': 'Team 3 info'
    },
    {
        'name': 'Team 4',
        'other': 'Team 4 info'
    }
]

leagues = [
    {
        'name': 'League 1'
    },
    {
        'name': 'League 2'
    },
    {
        'name': 'League 3'
    },
    {
        'name': 'League 4'
    },
    {
        'name': 'League 5'
    }
]

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', teams=teams, leagues=leagues)

@app.route("/about")
def about():
    return render_template('about.html', title="About")


host = '[HOST]'
port = 3306
user = '[USER]'
password = '[PASSWORD]'
dbname = '[DBNAME]'

try:
    print('Attempting to connect...')
    conn = pymysql.connect(host, user=user, port=port, passwd = password, db=dbname)
    print('Connected!')
    conn.close()
    print('Disconnected!')
except:
    print("An error occurred...")


# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
