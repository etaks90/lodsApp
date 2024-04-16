from flask import Flask, render_template, session, redirect, url_for, request, jsonify, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import logging, re, datetime, os
import pandas as pd
import numpy as np
from lib.nn import *
from lib.cb import *
# only load if not mocked. Loading takes a lot of time.
if os.getenv("openai__mock") == "0":
    from lib.docRec import *
from lib.utils import *
import matplotlib, tempfile
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from logging import LoggerAdapter
# https://stackoverflow.com/questions/69924881/userwarning-starting-a-matplotlib-gui-outside-of-the-main-thread-will-likely-fa
matplotlib.use('agg')

logger = create_logger()

list_session_parameter = ["delimiter", "map_col_names", "map_datatypes", "group_expressions", "table_connection", "new_cols"]

app = Flask(__name__)
# todo: save as env.
app.secret_key = 'your_secret_key'

###login stuff
login_manager = LoginManager()
login_manager.init_app(app)

login_manager.login_view = 'login'

# Dummy user data
users = {
    'oliver': {'password': 'oliver'},
    'user2': {'password': 'password2'}
}

# User class for Flask-Login
class User(UserMixin):
    pass



############################################################################################
####################GENERAL#######################################################
############################################################################################

@login_manager.user_loader
def load_user(user_id):
    user = User()
    user.id = user_id
    return user

@app.route('/', methods=['GET', 'POST'])
@log_url_call
def login():
    if request.method == 'POST':
        print("bbb")
        username = request.form['username']
        password = request.form['password']
        
        next_page = request.args.get('next')
        if username in users and users[username]['password'] == password:
            user = User()
            user.id = username
            login_user(user)
            if next_page is None:
                logger.debug(f"GO TO HOMEPAGE")
                return render_template('p__home.html')
            else:
                logger.debug(f"REDIRECT TO {next_page}")
                return redirect(next_page)
        else:
            return render_template('login.html', message='Invalid username or password.')
    else:
        return render_template('login.html')

@app.route('/p__home')
@log_url_call
@login_required
def p__home():
    return render_template('p__home.html')

@app.route('/logout')
@log_url_call
def logout():
    logout_user()
    return redirect(url_for('login'))
############################################################################################
####################REGRESSION/CLASSIFICATION#######################################################
############################################################################################

@app.route('/p__num__main')
@login_required
def p__num__main():
    clear_relevant_session()
    prepare_tmp_fp(current_user, "num")

    current_directory = os.getcwd()
    print("Current directory:", current_directory)

    logger.debug(f"SESSION: {session}")
    return render_template('p__num__main.html')

@app.route('/num__upload_tables', methods=['POST'])
def num__upload_tables():
    
    files = request.files.getlist('files[]')
    # Handle file upload logic here
    session["fp_input"] = []
    for file in files:
        logger.debug(f"FILE: {file.filename}")
        filename = secure_filename(file.filename)
        fp_tmp = os.path.join(session["fp_tmp_user_num"], filename)
        logger.debug(f"SAVE FILE AS {fp_tmp}")
        file.save(fp_tmp)

    response = jsonify({'message': 'Files uploaded successfully', 'filenames': session["fp_input"]})

    return response

# this fucntion can be called from multipel sources. Depending on source, different if conditions.
@app.route('/p__adapt_columnNames_columnTypes_set_join__numerical', methods=['POST'])
@login_required
def p__adapt_columnNames_columnTypes_set_join__numerical():
    log_page_call(list_session_parameter, "p__adapt_columnNames_columnTypes_set_join__numerical")
        
    # IF TRUE WE COME FROM FILE SELECTION. THEN WE:
    # 1.) SET SESSION PARAMETERS AND
    # 2.) SET DEFAULT VALUES (these value slater on are probably defined via AI).
    html_content__table_connection = ""
    if 'delimiter' in request.form.keys():
        logger.debug("FOLDER_PATH IN SESSION")
        #####SESSION PARAMETER##########
        # set new session-parameter
        for k in ["delimiter"]:
            session[k] = request.form[k]
        session["searchType"] = "grid"
        # remove old session-parameter
        for k in ["map_col_names", "map_datatypes", "group_expressions", "table_connection", "new_cols"]:
            session[k] = None

        # READ ORIGINAL DF AS IT IS NEEDED FOR DEFAULT FRONTEND PARAMETERS
        j__df__original = get_j__df(session['fp_tmp_user_num'], session['delimiter'], n__sample = 3)
        j__df = j__df__original.copy()
        textfields_frontend = get_frontend_default_columnNames_columnTypes_joins(j__df__original)
        session["new_cols"] = textfields_frontend["new_cols"]

        print(textfields_frontend["errors__table_connection"])

    else:
        logger.debug("FOLDER_PATH NOT IN SESSION")
        j__df__original = get_j__df(session['fp_tmp_user_num'], session['delimiter'], n__sample = 3)
        textfields_frontend = read_from_frontend_columnNames_columnTypes_joins()
        map__col_names, map__datatypes, textfields_frontend = parse_and_check_maps_from_frontend(j__df__original.copy(), textfields_frontend)
        # set new session-parameter
        for k in ["map_col_names", "map_datatypes", "group_expressions", "table_connection"]:
            session[k] = textfields_frontend[k]

        # Make copy of original table
        j__df = j__df__original.copy()

        # rename col names and adapt types
        j__df = adapt_cols_from_session(j__df, session)

        # GET HTML CONTENT FOR TABLE CONNECTION
        if len(textfields_frontend["table_connection"]) > 0:
            html_content__table_connection = vis_conn(textfields_frontend["table_connection"])
        else:
            html_content__table_connection = ""

    print(textfields_frontend)
    return render_template('p__adapt_columnNames_columnTypes_set_join__numerical.html'
                           , j__df__original = add_datatype(j__df__original), j__df__adapted = add_datatype(j__df), textfields_frontend = textfields_frontend
                           , html_content__table_connection = html_content__table_connection)

@app.route('/p__addNewColumns__numerical', methods=['POST'])
@login_required
def p__addNewColumns__numerical():
    logger.debug("START p__addNewColumns__numerical")

    # READ ORIGINAL DATA
    j__df = get_j__df(session['fp_tmp_user_num'], session['delimiter'])

    # RECALCULATE FROM SESSION
    j__df = adapt_cols_from_session(j__df, session)

    # JOIN
    m__table_connection = t_conn__str2map(session["table_connection"])
    name__main_table = get_name_main_table(m__table_connection, j__df)
    df__joined = join_dfs(j__df, name__main_table, m__table_connection)

    if "p__addNewColumns__numerical__button__continue" in request.form.keys():
        logger.debug("COMING FROM PREVIOUS PAGE p__adapt_columnNames_columnTypes_set_join__numerical VIA BUTTON")
        new_cols = session["new_cols"]
    else:
        logger.debug("NOT COMING FROM PREVIOUS PAGE p__adapt_columnNames_columnTypes_set_join__numerical VIA BUTTON")
        new_cols = request.form.get("additional_columns").split("\n")

    logger.debug(f"new_cols__string:\n{new_cols}")

    df__joined = add_new_cols(new_cols, df__joined.copy(), j__df.copy())

    return render_template('p__addNewColumns__numerical.html', available_columns = sorted(list(df__joined.columns))
                           , new_cols = new_cols, dataframe = df__joined.sample(min(5, len(df__joined))), len_df = len(df__joined), session = session)

@app.route('/p__setTrainParameter__numerical', methods=['POST'])
@login_required
def p__setTrainParameter__numerical():
    logger.debug("p__setTrainParameter__numerical")
    if "tar_col" in request.form:
        logger.debug(f"SET TARGET COLUMN")
        session["tar_col"] = request.form.get("tar_col").split("\n")[0]
    df = get_df(session)
    # get input cols
    df__num, df__cat = get_input_cols(df, session["tar_col"], th_num = 0, th_cat = 0)
    dont_include = ["default", "status (loan)", session["tar_col"]]
    num_cols = [e for e in df__num["Column Name (Numerical)"].to_list() if e not in dont_include]
    cat_cols = [e for e in df__cat["Column Name (Categorical)"].to_list() if e not in dont_include]
    df__num = df__num[~df__num["Column Name (Numerical)"].isin(dont_include)]
    df__cat = df__cat[~df__cat["Column Name (Categorical)"].isin(dont_include)]
    session["num_cols"] = num_cols
    session["num_cols_total"] = session["num_cols"]
    session["cat_cols"] = cat_cols
    session["cat_cols_total"] = session["cat_cols"]
    

    df__num["Type"] = "numerical"
    df__cat["Type"] = "categorical"

    d1 = df__num[["Column Name (Numerical)", "Correlation", "Type"]].copy()
    d1.columns = ["Column Name", "Correlation", "Type"]

    if df__cat.empty:
        print("d2 is empty#############")
        d2 = pd.DataFrame(columns=['a', 'b', 'c'])
    else:
        d2 = df__cat[["Column Name (Categorical)", "Correlation (mean)", "Type"]].copy()
        d2.columns = ["Column Name", "Correlation", "Type"]

    df_total = pd.concat([d1, d2])

    return render_template('p__setTrainParameter__numerical.html', df__num = df__num, df__cat = df__cat, data=df_total.to_dict('records'))

@app.route('/update_input_cols', methods=['POST'])
@login_required
def update_input_cols():
    logger.debug("update_input_cols")
    j = request.json
    print(j["data"])
    #selected_rows = j["data"][j["checkedRows"]]
    #print(selected_rows)
    selected_columns = []
    for i in j["checkedRows"]:
        selected_columns.append(j["data"][i][1])
    
    selected_columns = sorted(selected_columns)
    session["num_cols"] = [c for c in session["num_cols_total"] if c in selected_columns]
    session["cat_cols"] = [c for c in session["cat_cols_total"] if c in selected_columns]

    logger.debug(f"num_cols: {session['num_cols']}")
    logger.debug(f"cat_cols: {session['cat_cols']}")

    return ""
    

@app.route('/p__trainedNetwork__numerical', methods=['POST'])
@login_required
def p__trainedNetwork__numerical():
    logger.debug("p__trainedNetwork__numerical")
    if "name_network" in request.form:
        logger.debug(f"SET NAME_NETWORK COLUMN")
        session["name_network"] = re.sub(r"[^\w_]", "", request.form.get("name_network").split("\n")[0]) + "__" + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        logger.debug(session["name_network"])

    df = get_df(session)
    print(session.keys())
    # IMPORTANT TO PASS COPY TO NOT DO CHANGES!!!!
    df__num, df__cat = get_input_cols(df.copy(), session["tar_col"], th_num = 0, th_cat = 0)
    session["num_cols"] = [e for e in df__num["Column Name (Numerical)"].to_list() if e not in ["default", "status (loan)", session["tar_col"]]]
    session["cat_cols"] = [e for e in df__cat["Column Name (Categorical)"].to_list() if e not in ["default", "status (loan)", session["tar_col"]]]
    session["predict_type"] = set_classification_type(df, session)
    # Important to use pipeline 
    if session["predict_type"].lower() == "classification":
        list__methods=[RandomForestClassifier]
        session["main_metric"] = "f1_score"
        session["best_type"] = "MAX"
    else:
        list__methods=[LinearRegression, RandomForestRegressor, MLPRegressor]
        session["main_metric"] = "nmse"
        session["best_type"] = "MIN"

    pipeline, metr, result_visualization, res, res_with_best_metric = train_network(df, session, session["name_network"], list__methods, None)
    print(result_visualization)
    txt = f'BEST METRIC based on {session["main_metric"]} ({session["best_type"]}): {res_with_best_metric[0].__name__} with value {res_with_best_metric[1]["metric"]["test"][session["main_metric"]]}'
    print(session)
    return render_template('p__trainedNetwork__numerical.html', result_visualization=result_visualization, txt = txt, name_network = session["name_network"])


@app.route('/p__regClass__result', methods=['POST'])
@login_required
def p__regClass__result():
    logger.debug("p__regClass__result")
    original_relevant_columns = get_original_relevant_columns(session)
    final_columns = session["num_cols"] + session["cat_cols"]

    logger.debug(f"original_relevant_columns: {original_relevant_columns}")
    logger.debug(f"final_columns: {final_columns}")

    df__t1 = pd.DataFrame({'Column': original_relevant_columns})
    df__t1["Value"] = ""

    df__t2 = pd.DataFrame({'Column': final_columns})
    df__t2["Value"] = ""


    return render_template('p__regClass__result.html', data__t1=df__t1.to_dict('records'), data__t2=df__t2.to_dict('records'))

@app.route('/regClass__t1__result', methods=['POST'])
@login_required
def regClass__t1__result():
    print("regClass__t1__result")
    if request.method == 'POST':
        data = request.json  # Get the data sent from the frontend

        # Process the data as needed
        # In this example, we'll simply print the data
        print("Received data:", data)

        # Return a response (you can customize this according to your needs)
        response = "Data received successfully"
        return jsonify(response=response)
    else:
        return jsonify(error="Invalid request method")
    

@app.route('/regClass__t0__result', methods=['POST', "GET"])
@login_required
def regClass__t0__result():
    print("regClass__t0__result")
    fp = request.form.get('filename', 'data')
    print(fp)
    data = {
        'Name': ['John', 'Alice', 'Bob'],
        'Age': [25, 30, 35],
        'City': [50000, 60000, 70000]
    }

    df = pd.DataFrame(data)
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
        # Write the DataFrame to the temporary file
        df.to_excel(temp_file.name, index=False)

    
    return send_file(temp_file.name, as_attachment=True)

###############################################################################
# START: Select numerical network

@app.route('/p__num__selection', methods=['POST', "GET"])
@log_url_call
@login_required
def p__num__selection():   

    session["dummy_for_refresh"] = 1
    print(f"-----> {session}")
    return render_template('p__num__selection.html')

@app.route('/num__send_networks', methods=['POST', "GET"])
@log_url_call
@login_required
def num__send_networks():
    df = get_df_with_num_links(os.path.join("data", "num_networks"))
    data = df.to_dict(orient='records')

    return jsonify(data)

# End: Select numerical network
###############################################################################

###############################################################################
# START: go to specific network
@app.route('/p__num__specific/<name_network>', methods=['POST', "GET"])
@log_url_call
@login_required
def p__num__specific(name_network):
    # make same name as before
    prepare_tmp_fp(current_user, "num")
    session["fp_network"] = os.path.join("data", "num_networks", f"{name_network}.pickle")
    logger.debug(f"CURRENT FILEPATH OF TRAINED NETWORK: '{session['fp_network']}'")
    # READ FROM LOCAL NETWORK TO MAKE SURE ALL PARAMETERS ARE AVAILABLE
    session["name_network"] = name_network
    fp_network = os.path.join("data", "num_networks", f'{session["name_network"]}.pickle')
    logger.debug(f"READ LOCAL NETWORK {session['name_network']}")
    with open(fp_network, 'rb') as f:
            j = pickle.load(f)
    session.update(j["session"])
    

    return render_template('p__num__specific.html', name_network = session["name_network"], predict_type = session["predict_type"])

@app.route('/in__num_specific_upload_tables', methods=['POST'])
def in__num_specific_upload_tables():
    
    files = request.files.getlist('files[]')
    # Handle file upload logic here
    session["fp_input"] = []
    for file in files:
        logger.debug(f"FILE: {file.filename}")
        filename = secure_filename(file.filename)
        fp_tmp = os.path.join(session["fp_tmp_user_num"], filename)
        logger.debug(f"SAVE FILE AS {fp_tmp}")
        file.save(fp_tmp)

    response = jsonify({'message': 'Files uploaded successfully', 'filenames': session["fp_input"]})

    return response

@app.route('/out__num_specific_send_excel', methods=['POST', "GET"])
@log_url_call
def out__num_specific_send_excel():

    # read data
    df = get_df(session)

    # get pipeline
    with open(session["fp_network"], 'rb') as f:
            pipeline = pickle.load(f)["pipeline"]

    # use pipeline on data
    p = pipeline.predict(df)
    df["predicted"] = p

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
        # Write the DataFrame to the temporary file
        df.to_excel(temp_file.name, index=False)
        print(f"send it: {temp_file.name}")
        return send_file(temp_file.name, as_attachment=True)

# END: go to specific network
###############################################################################

############################################################################################
####################FAQ CHATBOT#######################################################
############################################################################################

@app.route('/p__cb__main')
@log_url_call
@login_required
def p__cb__main():
    if current_user.is_authenticated:
        print("Current user:", current_user.id)
    else:
        print("No user logged in")
    return render_template('p__cb__main.html')

@app.route('/p__cb__selection', methods=['POST', "GET"])
@log_url_call
@login_required
def p__cb__selection():
    if not CB in session:
        session[CB] = {}
        logger.debug(f"FAQ PATH INITIALIZED")
    
    if "cb__name" in request.form:
        cb__name = request.form["cb__name"]
        cb__faq = request.form["cb__faq"]
        with open(os.path.join("data", "chatbot_data", f"{cb__name}.txt"), "w") as f:
            f.write(cb__faq)
        session[CB][cb__name] = cb__faq
        print(f"-----> {session}")
        logger.debug(f"CREATE CHATBOT {cb__name} with faq {cb__faq[:20]}")
        print(f"-----> {session}")
    else:
        logger.debug("CALL WITHOUT CREATING NEW CHATBOT")

    session["dummy_for_refresh"] = 1
    print(f"-----> {session}")
    return render_template('p__cb__selection.html')

@app.route('/p__cb__specific/<chatbot_name>', methods=['POST', "GET"])
@log_url_call
@login_required
def p__cb__specific(chatbot_name):
    logger.debug(f"START p__cb__specific with parameter {chatbot_name}")
    
    return render_template('p__cb__specific.html', cb__name = chatbot_name)

#####################################################################################
# send df of available chatbots
@app.route('/send_available_chatbots', methods=['POST', "GET"])
@log_url_call
@login_required
def send_available_chatbots():
    df = get_df_with_chatbot_links(os.path.join("data", "chatbot_data"))
    data = df.to_dict(orient='records')

    return jsonify(data)

# send chatbot response
@app.route("/cb__response", methods=["POST"])
@login_required
def cb__response():
    cb__name = request.form["cb__name"]
    logger.debug(f"CALLED FOR cb__name: {cb__name}")
    # initialize history key
    if not (CBHISTORY in session):
        session[CBHISTORY] = {}
        logger.debug(f"HISTORY INITIALIZED")
    # initialize the chatbot with required information (e.g. FAQ).
    fp__faq = os.path.join("data", "chatbot_data", f"{cb__name}.txt")
    logger.debug(f"fp__faq: {fp__faq}")
    fp__cb_history = os.path.join("data", "chatbot_history", f"{cb__name}.json")
    logger.debug(f"fp__cb_history: {fp__cb_history}")
    if (os.path.exists(fp__cb_history)) and (1==1):
        with open(fp__cb_history, "r") as f:
            history = json.load(f)
        if len(history) > 6:
            history = history[:6]
            logger.debug(f"HISTORY TOO LONG: SHORTEN")
        logger.debug(f"HISTORY FOR {cb__name} ALREADY EXISTS.")
    # History does nto exists yet
    else:
        with open(fp__faq, "r") as f:
            faq = f.read()
        history = [{"role":"system","content":f"You are an AI assistant to answer questions based on this FAQ: '{faq}'."}]
        with open(fp__cb_history, "w") as f:
            json.dump(history, f)
        logger.debug(f"INITIALIZED HISTORY FOR CHATBOT {cb__name} in path {fp__faq}")        

    message = request.form["message"].lower()
    logger.debug(f"CB QUESTION: {message}")
    
    history = ask_question_cb(message, history, os.getenv("openai__engine"))

    return history[-1]["content"]

############################################################################################
####################DOCUMENTS#######################################################
############################################################################################

@app.route('/p__doc__main')
@log_url_call
@login_required
def p__doc__main():
    prepare_user_doc_filepath(current_user)

    return render_template('p__doc__main.html')

@app.route('/upload_images_doc', methods=['POST'])
def upload_images_doc():
    
    files = request.files.getlist('files[]')
    # Handle file upload logic here
    session["fp_input"] = []
    for file in files:
        logger.debug(f"FILE: {file.filename}")
        filename = secure_filename(file.filename)
        fp_tmp = os.path.join(session["fp_user_doc"], filename)
        logger.debug(f"SAVE FILE AS {fp_tmp}")
        file.save(fp_tmp)
        #session["fp_input"].append(filename)

    response = jsonify({'message': 'Files uploaded successfully', 'filenames': session["fp_input"]})

    return response

@app.route('/send_excel_doc', methods=['POST', "GET"])
@log_url_call
def send_excel_doc():
    # modules loads long, avoid reload
    logger.debug(f"MOCK STATUS, openai__mock: '{os.getenv('openai__mock')}'")
    logger.debug(f"EVAL IMAGES IN PATH {session['fp_user_doc']}")
    list_fields = [e.strip().replace("'", "").replace('"',"") for e in request.args['doc__fields'].split(",")]
    logger.debug(f"EVAL Fields {list_fields}")
    if os.getenv("openai__mock") == "0":
        logger.debug(f"REAL LLM RESPONSE")
        df = fpimg2df(session["fp_user_doc"], list_fields, os.getenv("openai__engine"), os.getenv("openai__mock"))
    else:
        logger.debug(f"MOCK LLM RESPONSE")
        df = pd.DataFrame()
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
        # Write the DataFrame to the temporary file
        df.to_excel(temp_file.name, index=False)
        print(f"send it: {temp_file.name}")
        return send_file(temp_file.name, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
