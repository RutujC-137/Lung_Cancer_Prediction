import pandas as pd
import mysql.connector
from mysql.connector import errorcode

# --- Configuration ---
# This is the correct path to your CSV file based on your 'dir' output.
csv_file_path = r'E:\Softwares\BDT Part1\Big data Staging area\Project\lung_cancer_project\preprocessed_lung_cancer_output\part-00000-215f17c3-3f9b-4132-9f8d-7253c27e8bf7-c000.csv'

db_config = {
    'host': 'localhost',
    'database': 'lung_cancer_db', # This should match the database you created in MySQL
    'user': 'root',               # Your MySQL username
    'password': 'root' # <<< IMPORTANT: REPLACE WITH YOUR ACTUAL MYSQL PASSWORD
}
table_name = 'preprocessed_lung_data' # Name of the table to create in MySQL

# --- Function to create table based on DataFrame schema ---
def create_table_from_df(cursor, df, table_name):
    # Mapping pandas dtypes to MySQL data types
    dtype_mapping = {
        'int64': 'INT',
        'float64': 'DOUBLE',
        'object': 'VARCHAR(255)', # For string columns, adjust length if needed
        'bool': 'BOOLEAN'
    }

    columns = []
    for col_name, dtype in df.dtypes.items():
        mysql_type = dtype_mapping.get(str(dtype), 'VARCHAR(255)') # Default to VARCHAR for unknown types
        # Handle specific column types if needed, e.g., for 'id' as PRIMARY KEY
        if col_name.lower() == 'id':
            columns.append(f"`{col_name}` {mysql_type} PRIMARY KEY")
        else:
            columns.append(f"`{col_name}` {mysql_type}")

    create_table_sql = f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(columns)})"
    print(f"Creating table with SQL: {create_table_sql}")
    cursor.execute(create_table_sql)
    print(f"Table `{table_name}` ensured to exist.")

# --- Main Script ---
try:
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Display first few rows and info for verification
    print("\nCSV Data Head:")
    print(df.head())
    print("\nCSV Data Info:")
    df.info()

    # Connect to MySQL
    print("\nConnecting to MySQL database...")
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()
    print("Successfully connected to MySQL.")

    # Create table based on DataFrame schema
    create_table_from_df(cursor, df, table_name)

    # Prepare for inserting data
    # Convert DataFrame to a list of tuples for insertion
    rows_to_insert = [tuple(row) for row in df.values]

    # Construct the INSERT statement dynamically
    # Use backticks for column names to avoid issues with reserved words
    columns_sql = ", ".join([f"`{col}`" for col in df.columns])
    placeholders = ", ".join(["%s"] * len(df.columns))
    insert_sql = f"INSERT INTO `{table_name}` ({columns_sql}) VALUES ({placeholders})"

    print(f"\nInserting {len(rows_to_insert)} rows into `{table_name}`...")
    # Insert all rows in one go using executemany for efficiency
    cursor.executemany(insert_sql, rows_to_insert)
    cnx.commit() # Commit the changes to the database

    print(f"Data successfully dumped into MySQL table `{table_name}`.")

except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Error: Access denied. Check your MySQL username and password.")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print(f"Error: Database '{db_config['database']}' does not exist. Please create it first.")
    else:
        print(f"MySQL Error: {err}")
except FileNotFoundError:
    print(f"Error: The CSV file was not found at {csv_file_path}. Please check the path and filename carefully.")
except pd.errors.EmptyDataError:
    print(f"Error: The CSV file at {csv_file_path} is empty or has no columns.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if 'cnx' in locals() and cnx.is_connected():
        cursor.close()
        cnx.close()
        print("MySQL connection closed.")