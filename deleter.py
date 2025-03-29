import sqlite3

# Connect to the database
conn = sqlite3.connect("db.sqlite3")
cursor = conn.cursor()

# Delete specific records


# Delete all records from a table
cursor.execute("DELETE FROM auth_user")

# Commit the changes and close the connection
conn.commit()
conn.close()