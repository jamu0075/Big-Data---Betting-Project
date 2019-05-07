# Sports-Betting - ATLAS 5214: Big Data Architecture

## Contributors

* Erik Johnson
* Nick Varberg
* Jacob Munoz

## Local Development
**_Make sure you create a branch to develop and DO NOT push to master. This will allow you to freely play around with the database without breaking code permanently_**
1. cd into **eb-btb-flask**
2. Activate the virtual environment to prevent version conflicts
```
source virt3/bin/activate
```
3. Download the required dependencies
```
pip install -r requirements.txt
```
4. Run the application
```
python application.py
```
> For now the database connection is hard coded and *not* included in the source. Replace the placeholder text in application.py to connect. 
```
host = '[HOST]'
port = 3306
user = '[USER]'
password = '[PASSWORD]'
dbname = '[DBNAME]'
```

## Project Summary

## Current Deployment Link
http://btbdeployment2-env.nmcjnh5dai.us-east-1.elasticbeanstalk.com/

## Datasets We Are Using

* [Kaggle European soccer data](https://www.kaggle.com/austro/beat-the-bookie-worldwide-football-dataset)
