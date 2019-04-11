from flask import Flask, render_template
import os
import pymysql

# EB looks for an 'app' callable by default.
app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')

'''
try:
    conn = pymysql.connect(host, user=user, port=port, passwd = password, db=dbname)
    print("Connected...")
except:
    print("An error occurred...")
'''

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
