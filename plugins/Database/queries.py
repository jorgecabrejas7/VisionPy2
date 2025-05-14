# imports
import psycopg2
import pandas as pd

# Create a connection to the PostgreSQL database
def connect(host, database, user, password):
    """
    Establishes a connection to the PostgreSQL database.

    Parameters:
    host (str): The database server address.
    database (str): The name of the database.
    user (str): The database user.
    password (str): The password for the database user.

    Returns:
    conn: A connection object to the PostgreSQL database.
    """
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password)
    return conn

# DATAFRAME MANAGEMENT

# Metadata concatenation
def metadata_add(data, metadata, id_column_name):
    """
    Adds metadata to the data dataframe.

    Parameters:
    data (DataFrame): The main data dataframe.
    metadata (DataFrame): The metadata dataframe.
    id_column_name (str): The column name in metadata that corresponds to the id in data.

    Returns:
    DataFrame: The data dataframe with metadata added.
    """
    # Get the id column of data
    ids = data['id']

    for id in ids:
        # Get the metadata where the id is equal to the value in id_column_name
        metadata_id = metadata[metadata[id_column_name] == id]

        # Iterate over the rows of the metadata_id
        for index, row in metadata_id.iterrows():
            column_name = row['key']
            value = row['value']
            units = row['type']
            data.loc[data['id'] == id, column_name] = str(value) + ' ' + units 

    return data

# Parent concatenation
def parent_add(data, parent_data, column_parent_id_name, suffixes=('_data', '_parent')):
    """
    Concatenates a parent dataframe to the main data dataframe.

    Parameters:
    data (DataFrame): The main data dataframe.
    parent_data (DataFrame): The parent data dataframe.
    column_parent_id_name (str): The column name in data that corresponds to the id in parent_data.
    suffixes (tuple): Suffixes to apply to overlapping column names in the left and right side, respectively.

    Returns:
    DataFrame: The concatenated dataframe.
    """

    merged_data = pd.merge(data, parent_data, left_on=column_parent_id_name, right_on='id' + suffixes[1], how='inner', suffixes = suffixes)

    # Drop the left_on column
    merged_data.drop(columns=[column_parent_id_name], inplace=True)

    return merged_data

# QUERY FUNCTIONS

# Table loading
def get_data(table_name, host, database, user, password):
    """
    Loads data from a specified table in the database.

    Parameters:
    table_name (str): The name of the table to load data from.
    host (str): The database server address.
    database (str): The name of the database.
    user (str): The database user.
    password (str): The password for the database user.

    Returns:
    DataFrame: The loaded data as a pandas dataframe.
    """
    ## First we connect to the database
    try:
        conn = connect(host,database,user,password)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    # Create a cursor object using the cursor() method
    cursor = conn.cursor()

    query = f"SELECT * FROM {table_name}"

    cursor.execute(query)

    # Fetch all the records
    records = cursor.fetchall()

    # Get the column names
    colnames = [desc[0] for desc in cursor.description]

    #create a pandas df

    data = pd.DataFrame(records, columns=colnames)

    #if a colum is full Nan, we drop it

    data = data.dropna(axis=1, how='all')

    #change the name of the columns to end with _table_name

    data.columns = [str(col) + '_' + table_name[:-1] for col in data.columns]

    # Close the cursor and the connection

    cursor.close()
    conn.close()

    return data

# Table + metadata loading
def get_data_metadata(table_name, host, database, user, password):
    """
    Loads data and its metadata from specified tables in the database.

    Parameters:
    table_name (str): The name of the table to load data from.
    host (str): The database server address.
    database (str): The name of the database.
    user (str): The database user.
    password (str): The password for the database user.

    Returns:
    DataFrame: The loaded data with metadata as a pandas dataframe.
    """
    metadata_name = table_name[:-1] + '_metadata'

    id_column_name = table_name[:-1] + '_id'

    ## First we connect to the database
    try:
        conn = connect(host,database,user,password)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    # Create a cursor object using the cursor() method
    cursor = conn.cursor()

    query = f"SELECT * FROM {table_name}"

    cursor.execute(query)

    # Fetch all the records
    records = cursor.fetchall()

    # Get the column names
    colnames = [desc[0] for desc in cursor.description]

    #create a pandas df

    data = pd.DataFrame(records, columns=colnames)

    #extract the metadata

    query = f"SELECT * FROM {metadata_name}"

    cursor.execute(query)

    # Fetch all the records

    records = cursor.fetchall()

    # Get the column names

    colnames = [desc[0] for desc in cursor.description]

    #create a pandas df

    metadata = pd.DataFrame(records, columns=colnames)

    #add metadata to the table

    data = metadata_add(data,metadata,id_column_name)

    #if a colum is full Nan, we drop it

    data = data.dropna(axis=1, how='all')

    #change the name of the columns to end with _table_name

    data.columns = [str(col) + '_' + table_name[:-1] for col in data.columns]

    # Close the cursor and the connection

    cursor.close()
    conn.close()

    return data

# Table + parent + metadata loading
def data_parent(table_name, parent_name, column_parent_id_name, host, database, user, password):
    """
    Loads data, its metadata, and parent data from specified tables in the database.

    Parameters:
    table_name (str): The name of the table to load data from.
    parent_name (str): The name of the parent table to load data from.
    column_parent_id_name (str): The column name in data that corresponds to the id in parent_data.

    Returns:
    DataFrame: The loaded data with metadata and parent data as a pandas dataframe.
    """
    data = get_data_metadata(table_name, host, database, user, password)

    parent_data = get_data_metadata(parent_name, host, database, user, password)

    column_parent_id_name = column_parent_id_name + '_' + table_name[:-1]

    suffixes = ('_' + table_name[:-1], '_' + parent_name[:-1])

    merged_data = parent_add(data,parent_data,column_parent_id_name,suffixes=suffixes)

    #if a colum is full Nan, we drop it

    merged_data = merged_data.dropna(axis=1, how='all')

    return merged_data

# Table + multiple parents + metadata loading
def multiple_parents(table_name, parents_names, column_parent_id_names, host, database, user, password):
    """
    Loads data, its metadata, and multiple parent data from specified tables in the database.

    Parameters:
    table_name (str): The name of the table to load data from.
    parents_names (list): The names of the parent tables to load data from.
    column_parent_id_names (list): The column names in data that correspond to the ids in parent_data.
    suffixes (list): Suffixes to apply to overlapping column names in the left and right side, respectively.

    Returns:
    DataFrame: The loaded data with metadata and multiple parent data as a pandas dataframe.
    """

    suffixes = [''] + ['_' + name[:-1] for name in parents_names]

    suffixes_list = []

    for i, value in enumerate(suffixes):
        if i == 0:
            suffixes_list.append([value, suffixes[i+1]])
        elif i > 1:
            suffixes_list.append(['', value])

    data = get_data_metadata(table_name, host, database, user, password)

    for parent_name,column_parent_id_name,suffixes in zip(parents_names,column_parent_id_names,suffixes_list):

        parent_data = get_data_metadata(parent_name, host, database, user, password)

        data = parent_add(data,parent_data,column_parent_id_name,suffixes=suffixes)
    

    #if a colum is full Nan, we drop it

    data = data.dropna(axis=1, how='all')
    
    return data

# Relation data loading
def relation_metadata(table1_name, table2_name, intermediate_table_name, host, database, user, password):
    """
    Loads data from two tables related by an intermediate relationship table.

    Parameters:
    table1_name (str): The name of the first table to load data from.
    table2_name (str): The name of the second table to load data from.
    column_id_1 (str): The column name in the first table that corresponds to the id in the intermediate table.
    column_id_2 (str): The column name in the second table that corresponds to the id in the intermediate table.
    intermediate_table_name (str): The name of the intermediate relationship table.
    suffixes (tuple): Suffixes to apply to overlapping column names in the left and right side, respectively.

    Returns:
    DataFrame: The loaded data with metadata from the two related tables as a pandas dataframe.
    """
    column_id_1 = table1_name[:-1] + '_id'

    column_id_2 = table2_name[:-1] + '_id'

    data1 = get_data_metadata(table1_name, host, database, user, password)

    data2 = get_data_metadata(table2_name, host, database, user, password)

    intermediate_data = get_data(intermediate_table_name, host, database, user, password)

    intermediate_data = intermediate_data.drop(columns=['id_'+intermediate_table_name[:-1]])

    suffixes = ('_' + table1_name[:-1], '_' + table2_name[:-1])

    #drop the id column of the intermediate data

    merged_data = pd.merge(data1, intermediate_data, left_on='id_' + table1_name[:-1], right_on=column_id_1 + '_' + intermediate_table_name[:-1], how='inner', suffixes=('', suffixes[0]))

    merged_data = pd.merge(merged_data, data2, left_on=column_id_2+ '_' + intermediate_table_name[:-1], right_on='id_' + table2_name[:-1], how='inner', suffixes=('', suffixes[1]))

    #drop all the columns from the intermediate table

    merged_data = merged_data.drop(columns=[c for c in intermediate_data.columns])

    #if a colum is full Nan, we drop it

    merged_data = merged_data.dropna(axis=1, how='all')

    return merged_data

