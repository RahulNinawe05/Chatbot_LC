import sqlite3

# connect to sqlite (database file banega folder me)
connection = sqlite3.connect(r"E:\1-Q&A Chatbot\Chatbot\5. Chat Sql Chatbot\class_db.db")

## create a cursor object to insert record , create table
cursor = connection.cursor()

# agar table pehle se hai to delete kar do
cursor.execute("DROP TABLE IF EXISTS STUDENT_TBL")

## create the table
table_info = """
CREATE TABLE STUDENT_TBL(
    NAME VARCHAR(25),
    CLASS VARCHAR(24),
    SECTION VARCHAR(25),
    MARKS INT
)
"""
cursor.execute(table_info)

## INSERT INTO SOME MORE RECORDS
cursor.execute("INSERT INTO STUDENT_TBL VALUES('Krish','Data Science','A',90)")
cursor.execute("INSERT INTO STUDENT_TBL VALUES('John','Data Science','B',100)")
cursor.execute("INSERT INTO STUDENT_TBL VALUES('Mukesh','Data Science','A',86)")
cursor.execute("INSERT INTO STUDENT_TBL VALUES('Jacob','DEVOPS','A',50)")
cursor.execute("INSERT INTO STUDENT_TBL VALUES('Dipesh','DEVOPS','A',35)")
cursor.execute("INSERT INTO STUDENT_TBL VALUES('RAJESH','WEB','A',87)")

## display all the records
data = cursor.execute("SELECT * FROM STUDENT_TBL")
for row in data:
    print(row)

## commit your changes in the db
connection.commit()
connection.close()
