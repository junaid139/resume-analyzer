import mysql.connector

def get_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Fardeen@123",
        database="resume_analyzer"
    )
    return connection