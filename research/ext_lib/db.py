"""Database-related stuff"""

import sqlalchemy
from sqlalchemy import create_engine, text

DB_USER = "postgres"
DB_PW = "KzsZ0wz5tp2wUIPM"
DB_HOST = "db.stratlab.dev"
DB_PORT = 30543
DB_URL = f"postgresql://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}"
print(DB_URL)
#DB_ARGS = {"sslmode": "verify-full", "sslrootcert": "system"}
DB_ARGS = {}

def db_connect(db_name: str) -> sqlalchemy.engine.Engine:
    """Connect to DB. Create it if it doesn't exist
    Args:
        db_name: name of the database to create
    """
    try:
        engine = create_engine(f"{DB_URL}/{db_name}", connect_args=DB_ARGS)
        engine.connect()
        return engine
    except sqlalchemy.exc.OperationalError:
        # Database doesn't exist, create it
        conn = create_engine(f"{DB_URL}/postgres", isolation_level="AUTOCOMMIT", connect_args=DB_ARGS).connect()
        # TODO: figure out how to get rid of SQL injection. Standard parameterization adds quotes that breaks syntax
        conn.execute(text(f"CREATE DATABASE {db_name}"))

        return create_engine(f"{DB_URL}/{db_name}", connect_args=DB_ARGS)


# list exchanges
# list symbols
# first date
# last date

# get data
# save data
