import sqlite3

connection = sqlite3.connect("pred.db")
c = connection.cursor()
c.execute(
    "CREATE TABLE plate (id varchar(10) NOT NULL PRIMARY KEY, plate_recognizer json, car_jam json, model json)"
)
connection.commit()
connection.close()
