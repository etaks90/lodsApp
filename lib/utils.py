from flask import Flask, render_template, session, redirect, url_for, request, jsonify
from flask_login import current_user
import logging, os, copy, re, pickle, time, random, uuid
import pandas as pd
import numpy as np
from pyvis.network import Network
from functools import wraps

# set private logger
class CurrentUserFilter(logging.Filter):
    """
    Add a user attribute to all records
    """
    def filter(self, record):
        try:
            record.remote_addr = request.remote_addr
        except:
            record.remote_addr = "offline"
        if current_user.is_authenticated:
            record.user = current_user.id
        else:
            record.user = "#unknown#"
        return True  # Return True to pass the record to the next filter

def create_logger(logger_name = 'logger'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(user)s -  %(remote_addr)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addFilter(CurrentUserFilter())

    return logger

logger = logging.getLogger('logger')

SESSIONID = "session_id"
CB = "cb"
CBHISTORY = "cb_history"
NUM = "num"

def log_url_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"CALLING ENDPOINT: {func.__name__}")
        result = func(*args, **kwargs)
        logger.debug(f"CALLED ENDPOINT: {func.__name__}")
        return result
    return wrapper

def dummy():
    print("dummy7")

def get_col_mapping_info():
    # todo: avoid -, +, [,... in col names.
    new__cols__full_description = {  "days_between" : {"syntax" : "days_between=date (loan)-date (account)", "type" : "minus", "columns" : "date (loan)____date (account)"}
                            , "av_unempl_rate" : {"syntax" : "av_unempl_rate=mean(unemployment rate '95 (district), unemployment rate '96 (district))", "type" : "mean", "columns" : "unemployment rate '95 (district)____unemployment rate '96 (district)"}
                            , "average_crime_rate_intermediate" : {"syntax" : "average_crime_rate_intermediate=mean(no. of commited crimes '95 (district), no. of commited crimes '96 (district))", "type" : "mean", "columns" : "no. of commited crimes '95 (district)____no. of commited crimes '96 (district)"}
                            , "average_crime_rate" : {"syntax" : "average_crime_rate=average_crime_rate_intermediate/no. of inhabitants (district)", "type" : "divide", "columns" : "average_crime_rate____no. of inhabitants (district)"}
                            , "default" : {"syntax" : "default=IF(status (loan);B,D;1;0)", "type" : "WHENTRUE__OR__B__D", "columns" : "status"}
                            , "owner age at opening" : {"syntax" : "owner age at opening=TIMEGAP(date (loan),birth_number (client),years)", "type" : "minus", "columns" : "date (loan)____birth_number (client)"}
                }
    return new__cols__full_description
        
def save_session(list_session_parameter):
    logger.debug(f"SAVE SESSION WITH THESE KEYS: {sorted(list(session.keys()))}")
    with open("session.pkl", 'wb') as f:
        pickle.dump({k:v for k, v in session.items() if k in list_session_parameter}, f)

def group_tables(j__df, group_calculations, append_tablename = True):
    df_new = {}
    if group_calculations:
        for table__group, v in group_calculations.items():
            if len(v) > 3:
                id_column = v.split("\n")[0]
                if append_tablename:
                    id_column = f"{id_column} ({table__group})"
                for group_info in v.split("\n")[1:]:
                    col_name = group_info.split(",")[2]
                    group_type = group_info.split(",")[1]
                    val_column = group_info.split(",")[0]
                    if append_tablename:
                        val_column = f"{val_column} ({table__group})"
                        col_name = f"{col_name} ({table__group})"
                    logger.debug(f"CREATE NEW COL '{col_name}' WITH VALUE COLUMN '{val_column}' grouped over '{id_column}' in table '{table__group}' AS '{group_type}'.")
                    df = j__df[table__group][[id_column, val_column]].copy()
                    if group_type == "mean":
                        df_grouped = df.groupby(id_column)[val_column].mean()
                    elif group_type == "max":
                        df_grouped = df.groupby(id_column)[val_column].max()
                    elif group_type == "min":
                        df_grouped = df.groupby(id_column)[val_column].min()
                    df_grouped = df_grouped.reset_index()
                    df_grouped.columns = [id_column, col_name]
                    try:
                        df_grouped[id_column] = df_grouped[id_column].astype(int)
                    except:
                        pass
                    if table__group in df_new:
                        df_new[table__group] = df_new[table__group].merge(df_grouped, on = id_column)
                    else:
                        df_new[table__group] = df_grouped

        for k in df_new:
            j__df[k] = df_new[k]

    return j__df

def get_j__df(fp__input_data, delimiter,n__sample = None):
    j__df = {}
    for root, dirs, files in os.walk(fp__input_data):
        for file in files:
            fp = os.path.join(root, file)
            # from mpg
            # https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
            df = pd.read_csv(fp, delimiter=delimiter, na_values='?', comment='\t', skipinitialspace=True, quotechar='"')
            df.columns = [e.replace("'", "").replace('"', "") for e in df.columns]
            fn = os.path.basename(fp).split(".")[0]
            if n__sample:
                j__df[fn] = df.sample(n=n__sample).copy()
            else:
                j__df[fn] = df.copy()

    return j__df

def clear_relevant_session(list_session_parameter = None):
    print(f"session: {session}")
    if list_session_parameter:
        logger.debug(f"CLEAR SESSION PARAMETERS: {'---'.join(list_session_parameter)}")
        for k in list_session_parameter:
            if k in session:
                del session[k]
    else:
        logger.debug(f"CLEAR ALL SESSION EXCEPT STARTING WITH '_'")
        # Get all keys in the session
        session_keys = list(session.keys())

        # Iterate through the keys and delete those not starting with '_'
        for key in session_keys:
            if not key.startswith('_'):
                del session[key]


def add_datatype(j__df_original, map__datatype_to_businessDatatype = {"int32" : "integer", "float32" : "float", "int64" : "integer", "float64" : "float", "object" : "string", "datetime64[ns]" : "date"}):
    """
    Deepcopy needed to not change original dict.
    """
    j__df = copy.deepcopy(j__df_original)
    for k, df in j__df.items():
        c = pd.MultiIndex.from_tuples([(col, header) for col, header in zip(df.columns, [str(c) for c in list(df.dtypes)])])
        c_new = []
        for e in c:
            e = (e[0], e[1], map__datatype_to_businessDatatype[e[1]])
            c_new.append(e)
        df.columns = c_new

    return j__df.copy()


def get_column_mapping(j):
    map__col = {}
    for key, value in j:
        if value == "":
            continue
        if key.startswith("columnMapping_"):
            table_name = key.replace("columnMapping_", "")
            map__col[table_name] = {v.split(";")[0] : v.split(";")[1] for v in value.split("\n")}

    return map__col

def adapt_column_names(j__df, map__col):
    for table_name, df in j__df.items():
        current_map = map__col.get(table_name)
        if current_map:
            logger.debug(f"FOR TABLE {table_name} MAP COLUMNS LIKE {current_map}")
            df = df.rename(columns=current_map)
            
            j__df[table_name] = df.copy()
    
    return j__df


def rps(input_string):
    return input_string.replace("\n","").replace("\r","")

def map__colnames_and_types_df(j__df, map__col_names, map__datatypes):
    j__df_mapped = {}
    for t, df in j__df.items():
            df_orig = df.copy()
            if map__col_names and (t in map__col_names):
                df = df.rename(columns=map__col_names[t])
            m__col2datatype__original_df = df.dtypes.astype(str).to_dict()
            for col_name, datatype__initial in m__col2datatype__original_df.items():
                dt__original = datatype__initial
                dt__new = None
                if map__datatypes and (t in map__datatypes) and (col_name in map__datatypes[t]):
                    dt__new = map__datatypes[t][col_name]
                else:
                    continue
                if not(dt__original == dt__new):
                    if dt__new.startswith("date"):
                        date_format = dt__new.split("(")[-1].replace(")","")
                        try:
                            df[col_name] = pd.to_datetime(df[col_name].astype(str), format=date_format)
                        except:
                            df.loc[df[col_name].astype(str).str[2:4].astype(int)>12,col_name] -= 5000
                            df[col_name] = pd.to_datetime("19" + df[col_name].astype(str), format='%Y%m%d')
                    elif dt__new.startswith("int"):
                        df[col_name] = df[col_name].astype(int)
                    elif dt__new.startswith("float"):
                        try:
                            df[col_name] = df[col_name].astype(float)
                        except:
                            # Filter out non-numeric values and calculate the mean
                            numeric_values = pd.to_numeric(df[col_name], errors='coerce')
                            numeric_values = numeric_values[~numeric_values.isna()]  # Filter out NaN values
                            average = numeric_values.mean()
                            # Replace the problematic values with the calculated average
                            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(average)

            j__df_mapped[t] = df

    return j__df_mapped

def vis_conn(t):
    rows = t.strip().split('\n')
    df_conn = pd.DataFrame([row.split(';') for row in rows])
    j__connections = {}
    for i, r in df_conn.iterrows():
        k = (r[0], r[1])
        v = ([r[2]], [r[3]])
        j__connections[k] = v

    # Example dictionary mapping tuples (t1, t2) to tuples ([c1], [c2])
    relationship_dict = j__connections

    # Create a network
    net = Network(notebook=True)

    # Add nodes and edges to the network
    for tables, columns in relationship_dict.items():
        net.add_node(tables[0], title=tables[0], type='table')
        net.add_node(tables[1], title=tables[1], type='table')
        net.add_edge(tables[0], tables[1], title=f"{columns[0][0]} ({tables[0]}) <-> {columns[1][0]}  ({tables[1]})")

    # the replacement is to make the style unusuable. Otherwise the style of teh whole page is affected. We want to avoid this!
    html_content__table_connection = net.generate_html().replace("stylesheet", "stylesheet123")

    return html_content__table_connection

def t_conn__str2map(t, split_str = ";"):
    if t:
        return {(rps(r.split(split_str)[0]), rps(r.split(split_str)[1])):(rps(r.split(split_str)[2]), rps(r.split(split_str)[3]), rps(r.split(split_str)[4])) for r in t.split("\n")}
    else:
        return None
    
def join_dfs(j__df, name__main_table, m__table_connection):
    logger.debug(f"JOIN DFS WITH MAIN TABLE {name__main_table} AND CONNECTIONS\n{m__table_connection}")
    # if only one entry do not join. Just return first entry.
    if len(j__df) == 1:
        return j__df[name__main_table]
    df__left = j__df[name__main_table]
    df__left.columns = [f"{e} ({name__main_table})" for e in df__left.columns]
    df__joined = df__left
    if 1 == 1:
        for join__tables, join__columns in m__table_connection.items():
            t__left = join__tables[0]
            t__right = join__tables[1]
            id__left = f"{join__columns[0]} ({t__left})"
            id__right = f"{join__columns[1]} ({t__right})"
            join_type = join__columns[2]

            df__right = j__df[t__right]
            df__right.columns = [f"{e} ({t__right})" for e in df__right.columns]

            logger.info(f"JOIN TABLE {t__left} with table {t__right} over IDs ({id__left} - {id__right}) AS {join_type}. Length Before Join: {len(df__joined)}")
            df__joined = df__joined.merge(df__right, left_on = id__left, right_on = id__right, how = join_type)
            logger.info(f"LENGTH AFTER JOIN {len(df__joined)}")
    return df__joined

def add_new_cols(new_cols__string, df__joined, session):
    # variable to determin which columsn depend on which
    j__col_dependencies = {}
    for e in new_cols__string:
        if e == "":
            continue
        col_name = e.split("=")[0].strip()
        logger.debug(e)
        t = e.split("=")[1].strip()
        logger.debug(col_name)
        if "+" in t:
            ee = [e.strip() for e in t.split("+")]
            logger.debug(f"SUM COLUMNS {ee} in new col {col_name}")
            df__joined[col_name] = df__joined[ee].sum(axis=1)
        elif "-" in t:
            ee = [e.strip() for e in t.split("-")]
            logger.debug(f"SUBSTRACT COLUMNS {ee} in new col {col_name}")
            try:
                df__joined[col_name] = df__joined[ee].apply(lambda row: row[ee[0]] - sum(row[ee[1:]]), axis=1)
            except:
                df__joined[col_name] = (df__joined[ee[0]] - df__joined[ee[1]]).dt.days
        elif "/" in t:
            ee = [e.strip() for e in t.split("/")]
            logger.debug(f"DIVIDE COLUMNS {ee} in new col {col_name}")
            df__joined[col_name] = df__joined[ee[0]] / df__joined[ee[1]]
        elif t.lower().startswith("mean("):
            t = t[5:-1]
            ee = [e.strip() for e in t.split(",")]
            logger.debug(f"TAKE MEAN OF COLUMNS {ee} in new col {col_name}")
            df__joined[col_name] = df__joined[ee].mean(axis=1)
        elif t.lower().startswith("if("):
            take_col = t.split(";")[0][3:]
            rel_values = t.split(";")[1].split(",")
            res_true = t.split(";")[2]
            res_false = t.split(";")[3][:-1]
            logger.debug(f"FOR COL {col_name} IF COL {take_col} IN {rel_values} then {res_true} else {res_false}")
            df__joined[col_name] = 0
            df__joined.loc[df__joined[take_col].isin(rel_values), col_name] = 1
            ee = list(take_col)
        j__col_dependencies[col_name] = ee
        

        logger.debug("SUCCESS")
    
    session["j__col_dependencies"] = j__col_dependencies

    return df__joined

def get_col_mapping_from_frontend():
    map__col_names__from_frontend = {}
    map__datatypes__from_frontend = {}
    map__group_calculations__from_frontend = {}
    m__table_connection__from_frontend = {}
    take_values_from_frontend = False
    # get values from frontend:
    for k in request.form.keys():
        table_name__from_textinput = k.split("__")[-1]
        if k.startswith("map_col_names"):
            take_values_from_frontend = True
            map__col_names__from_frontend[table_name__from_textinput] = {r.split(":")[0]:r.split(":")[1] for r in request.form[k].split("\r\n")}
        if k.startswith("map_datatypes"):
            take_values_from_frontend = True
            map__datatypes__from_frontend[table_name__from_textinput] = {rps(r.split(":")[0]):rps(":".join(r.split(":")[1:])) for r in request.form[k].split("\r\n")}
        if k.startswith("group_calculations__"):
            take_values_from_frontend = True
            map__group_calculations__from_frontend[table_name__from_textinput] = request.form[k].replace("\r", "")
        if k.startswith("table_connection"):
            take_values_from_frontend = True
            m__table_connection__from_frontend = t_conn__str2map(request.form[k])

    return map__col_names__from_frontend, map__datatypes__from_frontend, map__group_calculations__from_frontend, m__table_connection__from_frontend, take_values_from_frontend

def get_col_mapping_from_default(j__df):
    j__checks = {}
    m__table_connection__default = {}
    if "district" in j__df.keys():
        m__table_connection__default = {("loan","account"):("account_id","account_id", "inner"),
                                        ("account","district"):("district_id","district_id", "inner"),
                                        ("loan","trans"):("account_id","account_id", "inner"),
                                        ("loan","order"):("account_id","account_id", "inner"),
                                        ("loan","disp"):("account_id","account_id", "inner"),
                                        ("disp","card"):("disp_id","disp_id", "left"),
                                        ("disp","client"):("client_id","client_id", "inner")}
    default__map_col_names = {}
    default__map_datatypes = {}
    default__group_calculations = {}
    for t, df in j__df.items():
        # FIRST REPLACEMENT OF COL NAMES
        m__col_names = {c:c for c in df.columns}
        if t == "district":
            m__col_names = {
"A1":"district_id",
"A2":"district name",	
"A3":"region",	
"A4":"no. of inhabitants"	,
"A5":"no. of municipalities with inhabitants < 499"	,
"A6":"no. of municipalities with inhabitants 500-1999",	
"A7":"no. of municipalities with inhabitants 2000-9999"	,
"A8":"no. of municipalities with inhabitants >10000"	,
"A9":"no. of cities"	,
"A10":"ratio of urban inhabitants"	,
"A11":"average salary",
"A12":"unemployment rate '95",	
"A13":"unemployment rate '96",
"A14":"no. of enterpreneurs per 1000 inhabitants",
"A15":"no. of commited crimes '95",
"A16":"no. of commited crimes '96"
}                  
        
        # SECOND REPLACE DATATYPES BASED ON COL REPLACEMENTS
        df_dummy_for_datatypes = df.copy()
        df_dummy_for_datatypes = df_dummy_for_datatypes.rename(columns=m__col_names)
        m__datatypes = df_dummy_for_datatypes.dtypes.astype(str).to_dict()

        # map to default values
        if "birth_number" in m__datatypes:
            m__datatypes["birth_number"] = 'date (%y%m%d)'
        if "issued" in m__datatypes:
            m__datatypes["issued"] = 'date (%y%m%d %H:%M:%S)'
        if "date" in m__datatypes:
            m__datatypes["date"] = 'date (%y%m%d)'
        for col_name_rep in m__datatypes:
            if col_name_rep.startswith("no."):
                m__datatypes[col_name_rep] = "float64"
            elif col_name_rep.startswith("unemployment rate"):
                m__datatypes[col_name_rep] = "float64"
        
        if t == "order":
            default__group_calculations[t] = "account_id\namount,mean,mean_amt_order"
        elif t == "trans":
            default__group_calculations[t] = "account_id\namount,mean,mean_amt_trans\nbalance,mean,mean_blc_trans"
        else:
            default__group_calculations[t] = ""
        default__map_col_names[t] = m__col_names
        default__map_datatypes[t] = m__datatypes
        j__checks[f"map_col_names__{t}"] = []
        j__checks[f"map_datatypes__{t}"] = []

    return m__table_connection__default, default__map_col_names, default__map_datatypes, default__group_calculations, j__checks

def get_mapping(j__df):

    # GET MAPPING FROM FRONTEND
    map__col_names__from_frontend, map__datatypes__from_frontend, map__group_calculations__from_frontend, m__table_connection__from_frontend, take_values_from_frontend = get_col_mapping_from_frontend()
    
    # GET MAPPING FROM DEFAULT
    m__table_connection__default, default__map_col_names, default__map_datatypes, default__group_calculations, j__checks = get_col_mapping_from_default(j__df.copy())
    

    map__col_names = {}
    group_calculations = {}
    if take_values_from_frontend:
        map__col_names = map__col_names__from_frontend
        map__datatypes = map__datatypes__from_frontend
        group_calculations = map__group_calculations__from_frontend
        m__table_connection = m__table_connection__from_frontend
    else:
        map__col_names = default__map_col_names
        map__datatypes = default__map_datatypes
        group_calculations = default__group_calculations
        m__table_connection = m__table_connection__default

    return map__col_names, group_calculations, map__datatypes, m__table_connection, j__checks

def parse_maps(j__df, map__col_names, map__datatypes, j__checks):

    # 1.) ITERATE FOR DATATYPES
    for k in request.form.keys():
            table_name__from_textinput = k.split("__")[-1]
            if k.startswith("map_col_names"):
                textinput = request.form[k]
                if ":" in textinput:
                    map__col_names[table_name__from_textinput].update({rps(e.split(":")[0]) : rps(e.split(":")[1]) for e in textinput.split("\n")})
                col_names__from_df = [e.upper() for e in list(j__df[table_name__from_textinput].columns)]
                for mapping_source, mapping_target in map__col_names[table_name__from_textinput].items():
                    column_to_check = mapping_source.upper()
                    if column_to_check not in col_names__from_df:
                        j__checks[k].append(f"COLUMN {column_to_check} IS NOT CONTAINED IN COLUMNS.")

    # 2.) ITERATE FOR COL-NAMES
    for k in request.form.keys():
            table_name__from_textinput = k.split("__")[-1]
            if k.startswith("map_datatypes"):
                textinput = request.form[k]
                if ":" in textinput:
                    map__datatypes[table_name__from_textinput].update({rps(e.split(":")[0]) : rps(":".join(e.split(":")[1:])) for e in textinput.split("\n")})

    for k, v in j__checks.items():
        if not j__checks[k]:
            j__checks[k] = ""
        else:
            j__checks[k] = ";<br>".join(j__checks[k])

    return map__col_names, map__datatypes, j__checks

def get_maps_as_string(map__col_names, map__datatypes, m__table_connection):
    map__col_names__string = {}
    for k, v in map__col_names.items():
        map__col_names__string[k] = "\n".join([f"{kk}:{vv}" for kk, vv in v.items()])
    map__datatypes__string = {}
    for k, v in map__datatypes.items():
        map__datatypes__string[k] = "\n".join([f"{kk}:{vv}" for kk, vv in v.items()])

    m__table_connection__string = "\n".join([f"{k[0]};{k[1]};{v[0]};{v[1]};{v[2]}" for k, v in m__table_connection.items()])

    return map__col_names__string, map__datatypes__string, m__table_connection__string

def get_errors_in_connection_field(m__table_connection, j__df_mapped):
    errors__table_connection = []
    for k, v in m__table_connection.items():
        available_tables = j__df_mapped.keys()
        if k[0] not in available_tables:
            errors__table_connection.append(f"TABLE {k[0]} DOES NOT EXIST.")
        if k[1] not in available_tables:
            errors__table_connection.append(f"TABLE {k[1]} DOES NOT EXIST.")
        if (k[0] in available_tables) and not(v[0] in j__df_mapped[k[0]].columns):
            errors__table_connection.append(f"COLUMN {v[0]} DOES NOT EXIST IN TABLE {k[0]}.")
        if (k[1] in available_tables) and not(v[1] in j__df_mapped[k[1]].columns):
            errors__table_connection.append(f"COLUMN {v[1]} DOES NOT EXIST IN TABLE {k[1]}.")
    
    errors__table_connection = ";<br>".join(errors__table_connection)

    return errors__table_connection

def get_frontend_default_columnNames_columnTypes_joins(j__df):
    frontend_default = {}
    frontend_default["table_connection"] = {}
    frontend_default["map_col_names"] = {}
    frontend_default["map_datatypes"] = {}
    frontend_default["group_expressions"] = {}
    frontend_default["checks__colRenaming"] = {}
    frontend_default["checks__datatypes"] = {}
    frontend_default["errors__table_connection"] = ""

    # DEFAULT FOR MAPPING TABLE CONNECTIONS
    if "district" in j__df.keys():
        frontend_default["table_connection"] = """loan;account;account_id;account_id;inner
account;district;district_id;district_id;inner
loan;trans;account_id;account_id;inner
loan;order;account_id;account_id;inner
loan;disp;account_id;account_id;inner
disp;card;disp_id;disp_id;left
disp;client;client_id;client_id;inner"""
    
    # DEFAULT FOR COLUMN NAMES
    # Do first as other defaults depend on thus
    for t, df in j__df.items():
        # FIRST REPLACEMENT OF COL NAMES
        frontend_default["map_col_names"][t] = "\n".join(f"{c}:{c}" for c in df.columns)
        if t == "district":                
            frontend_default["map_col_names"][t] = """A1:district_id
A2:district name
A3:region
A4:no. of inhabitants
A5:no. of municipalities with inhabitants < 499
A6:no. of municipalities with inhabitants 500-1999
A7:no. of municipalities with inhabitants 2000-9999
A8:no. of municipalities with inhabitants >10000
A9:no. of cities
A10:ratio of urban inhabitants
A11:average salary
A12:unemployment rate '95
A13:unemployment rate '96
A14:no. of enterpreneurs per 1000 inhabitants
A15:no. of commited crimes '95
A16:no. of commited crimes '96"""

    # GET NEW COL NAMES
    map_col_names = {}
    for table_name, txt in frontend_default["map_col_names"].items():
        map_col_names[table_name] = {v.split(":")[0] : v.split(":")[1] for v in txt.split("\n")}

    # MAP FOR NEXT STEPS
    j__df = {t : df.rename(columns=map_col_names[t]) for t, df in j__df.items()}

    # DEFAULT OTHERS
    for t, df in j__df.items():
        current_map_datatypes = {k : v for k, v in df.dtypes.astype(str).to_dict().items()}
        

        # map to default values
        if "birth_number" in current_map_datatypes:
            current_map_datatypes["birth_number"] = 'date (%y%m%d)'
        if "issued" in current_map_datatypes:
            current_map_datatypes["issued"] = 'date (%y%m%d %H:%M:%S)'
        if "date" in current_map_datatypes:
            current_map_datatypes["date"] = 'date (%y%m%d)'
        for col_name_rep in current_map_datatypes:
            if col_name_rep.startswith("no."):
                current_map_datatypes[col_name_rep] = "float64"
            elif col_name_rep.startswith("unemployment rate"):
                current_map_datatypes[col_name_rep] = "float64"

        frontend_default["map_datatypes"][t] = "\n".join([f"{k}:{v}" for k, v in current_map_datatypes.items()])

        if t == "order":
            frontend_default["group_expressions"][t] = "account_id\namount,mean,mean_amt_order"
        elif t == "trans":
            frontend_default["group_expressions"][t] = "account_id\namount,mean,mean_amt_trans\nbalance,mean,mean_blc_trans"
        else:
            frontend_default["group_expressions"][t] = ""
        frontend_default["checks__colRenaming"][t] = ""
        frontend_default["checks__datatypes"][t] = ""


    if ("account" in j__df):
        new__cols__full_description = get_col_mapping_info()
        frontend_default["new_cols"] = [v["syntax"] + "\n" for k, v in new__cols__full_description.items()]
    else:
        frontend_default["new_cols"] = []
    

    return frontend_default

def read_from_frontend_columnNames_columnTypes_joins():
    text_from_frontend = {}
    text_from_frontend["table_connection"] = {}
    text_from_frontend["map_col_names"] = {}
    text_from_frontend["map_datatypes"] = {}
    text_from_frontend["group_expressions"] = {}
    # get values from frontend:
    for k in request.form.keys():
        table_name__from_textinput = k.split("__")[-1]
        if k.startswith("map_col_names"):
            text_from_frontend["map_col_names"][table_name__from_textinput] = request.form[k]
        if k.startswith("map_datatypes"):
            text_from_frontend["map_datatypes"][table_name__from_textinput] = request.form[k]
        if k.startswith("group_calculations__"):
            text_from_frontend["group_expressions"][table_name__from_textinput] = request.form[k].replace("\r","")
        if k.startswith("table_connection"):
            text_from_frontend["table_connection"] = request.form[k]

    return text_from_frontend


def log_page_call(list_session_parameter, name_endpoint):
    l = []
    for k in list_session_parameter:
        if k in session:
            l.append(f"{k} --> {session[k]}")
    logger.debug(f"CALLED ENDPOINT {name_endpoint} WITH PARAMETERS\n{'----'.join(l)}")

def str2map(str_type, txt):
    if str_type == "map_datatypes":
        return {rps(e.split(":")[0]) : rps(":".join(e.split(":")[1:])) for e in txt.split("\n")}
    elif str_type == "map_col_names":
        return {rps(e.split(":")[0]) : rps(e.split(":")[1]) for e in txt.split("\n")}
    elif str_type == "table_connection":
        return {(k.split(";")[0], k.split(";")[1]) : (k.split(";")[2], k.split(";")[3], k.split(";")[4]) for k in txt.split("\n")}

def parse_and_check_maps_from_frontend(j__df, textfields_frontend):

    map__col_names = {}
    map__datatypes = {}
    textfields_frontend["checks__colRenaming"] = {}
    textfields_frontend["checks__datatypes"] = {}

    check_colnames = True
    # 1.) 
    for table_name__from_textinput, text_frontend in textfields_frontend["map_col_names"].items():
        textfields_frontend["checks__colRenaming"][table_name__from_textinput] = []
        if ":" in text_frontend:
            map__col_names[table_name__from_textinput] = str2map("map_col_names", text_frontend)
        col_names__from_df = [e.upper() for e in list(j__df[table_name__from_textinput].columns)]
        for mapping_source, mapping_target in map__col_names[table_name__from_textinput].items():
            column_to_check = mapping_source.upper()
            if column_to_check not in col_names__from_df:
                textfields_frontend["checks__colRenaming"][table_name__from_textinput].append(f"COLUMN {column_to_check} IS NOT CONTAINED IN COLUMNS.")
                check_colnames = False

    # if checkk successful then rename columns
    if check_colnames:
        for k in j__df:
            j__df[k] = j__df[k].rename(columns = map__col_names[k])

    for table_name__from_textinput, text_frontend in textfields_frontend["map_datatypes"].items():
        if ":" in text_frontend:
            map__datatypes[table_name__from_textinput] = str2map("map_datatypes", text_frontend)

    if len(textfields_frontend["table_connection"]) > 0:
        # 3.) check table connections
        map__table_connection = str2map("table_connection", textfields_frontend["table_connection"])
        errors__table_connection = []
        for k, v in map__table_connection.items():
            available_tables = j__df.keys()
            if k[0] not in available_tables:
                print("aaa")
                errors__table_connection.append(f"TABLE {k[0]} DOES NOT EXIST.")
            if k[1] not in available_tables:
                errors__table_connection.append(f"TABLE {k[1]} DOES NOT EXIST.")
            if (k[0] in available_tables) and not(v[0] in j__df[k[0]].columns):
                errors__table_connection.append(f"COLUMN {v[0]} DOES NOT EXIST IN TABLE {k[0]}.")
            if (k[1] in available_tables) and not(v[1] in j__df[k[1]].columns):
                errors__table_connection.append(f"COLUMN {v[1]} DOES NOT EXIST IN TABLE {k[1]}.")
        
        if errors__table_connection:
            errors__table_connection = ";<br>".join(errors__table_connection)
        else:
            errors__table_connection = ""
            
        textfields_frontend["errors__table_connection"] = errors__table_connection
        print(f"errors__table_connection. {errors__table_connection}")
    else:
        textfields_frontend["errors__table_connection"] = ""
    print("aaaaaaaaaaa")
    for k, v in j__df.items():
        if not (k in textfields_frontend["checks__colRenaming"]):
            textfields_frontend["checks__colRenaming"][k] = ""
        else:
            textfields_frontend["checks__colRenaming"][k] = ";<br>".join(textfields_frontend["checks__colRenaming"][k])

        if not (k in textfields_frontend["checks__datatypes"]):
            textfields_frontend["checks__datatypes"][k] = ""
        else:
            textfields_frontend["checks__datatypes"][k] = ";<br>".join(textfields_frontend["checks__datatypes"][k])

    

    return map__col_names, map__datatypes, textfields_frontend

def adapt_cols_from_session(j__df, session, append_tablename = False):

    map__col_names = {t : str2map("map_col_names", session["map_col_names"][t]) for t in j__df.keys()}
    map__datatypes = {t : str2map("map_datatypes", session["map_datatypes"][t]) for t in j__df.keys()}
    group_calculations = session["group_expressions"]

    j__df_mapped = {}
    for t, df in j__df.items():
            df_orig = df.copy()
            if map__col_names and (t in map__col_names):
                df = df.rename(columns=map__col_names[t])
            m__col2datatype__original_df = df.dtypes.astype(str).to_dict()
            for col_name, datatype__initial in m__col2datatype__original_df.items():
                dt__original = datatype__initial
                dt__new = None
                if map__datatypes and (t in map__datatypes) and (col_name in map__datatypes[t]):
                    dt__new = map__datatypes[t][col_name]
                else:
                    continue
                if not(dt__original == dt__new):
                    if dt__new.startswith("date"):
                        date_format = dt__new.split("(")[-1].replace(")","")
                        try:
                            df[col_name] = pd.to_datetime(df[col_name].astype(str), format=date_format)
                        except:
                            df.loc[df[col_name].astype(str).str[2:4].astype(int)>12,col_name] -= 5000
                            df[col_name] = pd.to_datetime("19" + df[col_name].astype(str), format='%Y%m%d')
                    elif dt__new.startswith("int"):
                        df[col_name] = df[col_name].astype(int)
                    elif dt__new.startswith("float"):
                        try:
                            df[col_name] = df[col_name].astype(float)
                        except:
                            # Filter out non-numeric values and calculate the mean
                            numeric_values = pd.to_numeric(df[col_name], errors='coerce')
                            numeric_values = numeric_values[~numeric_values.isna()]  # Filter out NaN values
                            average = numeric_values.mean()
                            # Replace the problematic values with the calculated average
                            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(average)

            j__df_mapped[t] = df

    j__df = j__df_mapped.copy()

    df_new = {}
    if group_calculations:
        for table__group, v in group_calculations.items():
            if len(v) > 3:
                id_column = v.split("\n")[0]
                if append_tablename:
                    id_column = f"{id_column} ({table__group})"
                for group_info in v.split("\n")[1:]:
                    col_name = group_info.split(",")[2]
                    group_type = group_info.split(",")[1]
                    val_column = group_info.split(",")[0]
                    if append_tablename:
                        val_column = f"{val_column} ({table__group})"
                        col_name = f"{col_name} ({table__group})"
                    logger.debug(f"CREATE NEW COL '{col_name}' WITH VALUE COLUMN '{val_column}' grouped over '{id_column}' in table '{table__group}' AS '{group_type}'.")
                    df = j__df[table__group][[id_column, val_column]].copy()
                    if group_type == "mean":
                        df_grouped = df.groupby(id_column)[val_column].mean()
                    elif group_type == "max":
                        df_grouped = df.groupby(id_column)[val_column].max()
                    elif group_type == "min":
                        df_grouped = df.groupby(id_column)[val_column].min()
                    df_grouped = df_grouped.reset_index()
                    df_grouped.columns = [id_column, col_name]
                    try:
                        df_grouped[id_column] = df_grouped[id_column].astype(int)
                    except:
                        pass
                    if table__group in df_new:
                        df_new[table__group] = df_new[table__group].merge(df_grouped, on = id_column)
                    else:
                        df_new[table__group] = df_grouped

        for k in df_new:
            j__df[k] = df_new[k]

    return j__df

def has_non_integer_float(lst):
    for item in lst:
        if isinstance(item, float) and not item.is_integer():
            return True
    return False

def get_original_relevant_columns(session):
    input_columns = session["cat_cols"] + session["num_cols"]
    original_relevant_cols = input_columns.copy()
    ct = 1
    while True:
        leave_loop = True
        ct = ct + 1
        if ct > 100:
            break
        for c in original_relevant_cols:
            if c in session["j__col_dependencies"]:
                leave_loop = False
                original_relevant_cols = original_relevant_cols + session["j__col_dependencies"][c]
                original_relevant_cols.remove(c)

        if leave_loop:
            break

    return sorted(list(set(original_relevant_cols)))

def random_key():
    timestamp = int(time.time() * 1000)  # Convert current time to milliseconds
    random_number = random.randint(0, 1000)
    return f"{timestamp}-{random_number}"

def set_session_id():
    if not SESSIONID in session:
        session[SESSIONID] = uuid.uuid4()

def get_files_fullPath_in_directory(fp):
    files = os.listdir(fp)
    return [os.path.join(fp, fn) for fn in files]

def get_df_with_chatbot_links(fp):

    l__fp = get_files_fullPath_in_directory(fp)

    l__nw = [e.split("\\")[-1].split(".")[0] for e in l__fp]

    l__links = [f'<a href="/p__cb__specific/{e}">{e}</a>' for e in l__nw]

    return pd.DataFrame({"NAME NETWORK": l__nw, "LINK NETWORK": l__links})

def get_df_with_num_links(fp):

    l__fp = get_files_fullPath_in_directory(fp)

    l__nw = [e.split("\\")[-1].split(".")[0] for e in l__fp]

    l__links = [f'<a href="/p__num__specific/{e}">{e}</a>' for e in l__nw]

    return pd.DataFrame({"NAME NETWORK": l__nw, "LINK NETWORK": l__links})

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def prepare_user_doc_filepath(current_user):
    session["fp_user_doc"] = os.path.join("data", "tmp__doc", current_user.id)
    os.makedirs(session["fp_user_doc"], exist_ok=True)
    delete_files_in_folder(session["fp_user_doc"])
    logger.debug(f"FIELPATH {session['fp_user_doc']} EXISTS AND IS EMPTY.")

def prepare_tmp_fp(current_user, fp__type):
    fp_user_tmp = f"fp_tmp_user_{fp__type}"
    session[fp_user_tmp] = os.path.join("data", f"tmp__{fp__type}", current_user.id)
    os.makedirs(session[fp_user_tmp], exist_ok=True)
    delete_files_in_folder(session[fp_user_tmp])
    logger.debug(f"FIELPATH {session[fp_user_tmp]} EXISTS WITH KEY {fp_user_tmp} AND IS EMPTY.")