import os
import sqlalchemy 
from flask import Flask

db_user = os.environ.get('MYSQL_USER')
db_password = os.environ.get('MYSQL_PASSWORD')
db_name = os.environ.get('MYSQL_NAME')
db_connenction_name = os.environ.get('MYSQL_DSN')

#FROM GCP TUTORIALS: https://cloud.google.com/appengine/docs/standard/python3/using-cloud-sql
# When deployed to App Engine, the `GAE_ENV` environment variable will be
# set to `standard`
if os.environ.get('GAE_ENV') == 'standard':
    # If deployed, use the local socket interface for accessing Cloud SQL
    unix_socket = '/cloudsql/{}'.format(db_connection_name)
    engine_url = 'mysql+pymysql://{}:{}@/{}?unix_socket={}'.format(
        db_user, db_password, db_name, unix_socket)
else:
    # If running locally, use the TCP connections instead
    # Set up Cloud SQL Proxy (cloud.google.com/sql/docs/mysql/sql-proxy)
    # so that your application can use 127.0.0.1:3306 to connect to your
    # Cloud SQL instance
    host = '127.0.0.1'
    engine_url = 'mysql+pymysql://{}:{}@{}/{}'.format(
        db_user, db_password, host, db_name)

engine = sqlalchemy.create_engine(engine_url, pool_size=3)
app = Flask(__name__)

@app.route('/')
def hello():
    #conn = engine.connect()

    #conn.close()
    return 'Hello World!!'

if __name__ == '__main__':
    app.run(debug=True)