import time

from sqlalchemy import create_engine, inspect, text
import pandas as pd
import re
from urllib.parse import quote
import os
class DataBase():
    def __init__(self):
        # Allow overrides via env vars for cluster usage.
        self.db_user = os.getenv("TRACEBACK_MYSQL_USER", "root")
        self.db_password = quote(os.getenv("TRACEBACK_MYSQL_PASSWORD", ""))
        self.db_host = os.getenv("TRACEBACK_MYSQL_HOST", "localhost")
        self.db_port = int(os.getenv("TRACEBACK_MYSQL_PORT", "3307"))
        self.db_name = os.getenv("TRACEBACK_MYSQL_DB", "text2sql")

        # Explicitly include port so we connect to the user-local MySQL
        # instance started on 3307 (not the system default 3306).
        self.engine = create_engine(
            f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )

        self.inspector = inspect(self.engine)

    def __getstate__(self):
        # exclude engine from being pickled
        state = self.__dict__.copy()
        if 'engine' in state:
            del state['engine']
        return state

    def __setstate__(self, state):
        # restore attributes
        self.__dict__.update(state)
        # re-create the engine after unpickling
        self.db_user = os.getenv("TRACEBACK_MYSQL_USER", getattr(self, "db_user", "root"))
        self.db_password = quote(os.getenv("TRACEBACK_MYSQL_PASSWORD", ""))
        self.db_host = os.getenv("TRACEBACK_MYSQL_HOST", getattr(self, "db_host", "localhost"))
        self.db_port = int(os.getenv("TRACEBACK_MYSQL_PORT", str(getattr(self, "db_port", 3307))))
        self.db_name = os.getenv("TRACEBACK_MYSQL_DB", getattr(self, "db_name", "text2sql"))
        self.engine = create_engine(
            f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        )
    def table_not_exist(self,table_name):
        return table_name not in self.inspector.get_table_names()

    def upload_table(self, table_name, df):
        try:
            df.to_sql(table_name, con=self.engine, if_exists="replace", index=False)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)

    def close_connection(self):
        self.engine.dispose()

    def run_sql(self, query):
        with self.engine.connect() as connection:
            # Make LLM-generated SQL more MySQL-compatible:
            # - Accept identifiers quoted with double-quotes when ANSI_QUOTES is enabled.
            try:
                connection.execute(text("SET SESSION sql_mode = CONCAT(@@sql_mode, ',ANSI_QUOTES')"))
            except Exception:
                pass

            query = (query or "").strip()
            # Some prompts use "<SQL>: ..." so strip a leading ':' or 'SQL:' if present.
            if query.startswith(":"):
                query = query.lstrip(":").strip()
            if query.upper().startswith("SQL:"):
                query = query[4:].strip()

            query_type = query.strip().split()[0].upper()
            if query_type == 'SELECT':
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
            elif query_type == 'CREATE':
                pattern = r'(?i)\bCREATE\s+TABLE\s+[`"]?([\w]+(?:\.[\w]+)?)[`"]?'
                match = re.search(pattern, query)
                if match:
                    table_name = match.group(1)
                else:
                    print(f'Table name cannot be extracted from: \n{query}\n')
                    return
                table_name = table_name.lower()

                connection.execute(text(f"DROP TABLE IF EXISTS `{table_name}`"))
                time.sleep(1)
                result = connection.execute(text(query))
                connection.commit()
                query = f'Select * from {table_name}'
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
            else:
                print(f'Not Select/Create Query: {query_type}')
                return None

    def get_all_rows(self, table_name):
        with self.engine.connect() as connection:
            query = f'Select * from {table_name}'
            result = connection.execute(text(query))
            connection.commit()


            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
