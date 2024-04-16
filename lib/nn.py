from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import pickle, logging, io, datetime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from io import BytesIO
import base64
from lib.utils import *

logger = logging.getLogger('logger')

def calculate_correlation_for_categorical_columns(df__joined, cat_columns, target_col):

    # Assuming df is your DataFrame
    # Assuming l is your list of string columns
    # Assuming d is your numeric column

    # Convert string columns to categorical
    cat_columns = df__joined.select_dtypes(include=['object']).columns.tolist()
    categorical_df = df__joined[cat_columns].astype('category')

    # Perform one-hot encoding
    encoded_df = pd.get_dummies(categorical_df)

    # Concatenate with numeric column
    combined_df = pd.concat([encoded_df, df__joined[target_col]], axis=1)

    # Calculate correlation coefficients
    correlations = combined_df.corr()

    # Print correlation coefficients
    correlations["index"] = correlations.index.str.split("_").str[0]
    # Calculate the absolute values of correlations
    # Drop column "d" before calculating absolute values
    correlations_without_d = correlations.drop(columns=["index"])

    # Calculate the absolute values of correlations
    absolute_correlations_without_d = correlations_without_d.abs()

    # Concatenate "d" back to the DataFrame
    absolute_correlations = pd.concat([absolute_correlations_without_d, correlations["index"]], axis=1)


    # Group by column index and calculate mean, median, and count
    grouped_statistics = absolute_correlations.groupby(by="index").agg(['mean', 'median', 'count'])

    # Print grouped statistics
    grouped_statistics = grouped_statistics.reset_index()
    df = grouped_statistics[["index", target_col]]

    return df

def get_num_cols(df__joined, target_col, th_num):
    numeric_columns = df__joined.select_dtypes(include=['int', 'float']).columns.tolist()
    if df__joined[target_col].dtype == 'object':
        logger.debug("TARGET COL IS CAT")
        label_encoder = LabelEncoder()
        df__joined[target_col] = label_encoder.fit_transform(df__joined[target_col])
        correlations = df__joined[numeric_columns].corrwith(df__joined[target_col])
    else:
        correlations = df__joined[numeric_columns].corrwith(df__joined[target_col])
    num_cols__default = correlations.abs().sort_values(ascending=False)
    df_num = pd.DataFrame({'Column Name (Numerical)': num_cols__default.index, 'Correlation': num_cols__default.values})
    return df_num
    num_cols = [e for e in list(num_cols__default[num_cols__default>th_num].index) if e not in ["default", "status (loan)"]]

    return num_cols

def get_cat_cols(df__joined, target_col, th_cat):
    cat_columns = df__joined.select_dtypes(include=['object']).columns.tolist()
    if len(cat_columns) == 0:
        logger.debug("NO CATEGORICAL COLUMNS")
        return pd.DataFrame(columns=['Column Name (Categorical)'])
    df__corr_cat = calculate_correlation_for_categorical_columns(df__joined.copy(), cat_columns, target_col)
    df = df__corr_cat.reset_index()
    df = df__corr_cat.droplevel(0, axis=1)
    df.columns = ["index", "mean", "median", "count"]
    df__cat = df.sort_values(by='mean', ascending=False)
    df__cat = df__cat.rename(columns={'index': 'Column Name (Categorical)', 'mean': 'Correlation (mean)', 'median': 'Correlation (median)', 'count': 'Number generated fields'})
    # only use categorical columns
    df__cat = df__cat[df__cat['Column Name (Categorical)'].isin(cat_columns)]
    return df__cat
    cat_cols = [e for e in df.loc[df['mean'] > th_cat, 'index'] if e not in ["default", "status (loan)"]]

    return cat_cols

def get_input_cols(df__joined, target_col, th_num, th_cat):
    num_cols = get_num_cols(df__joined, target_col, th_num)
    cat_cols = get_cat_cols(df__joined, target_col, th_cat)

    return num_cols, cat_cols

def test_pipeline_with_full_df(pipeline, df, session):
    j = []
    for i in range(len(df)):
        j_dummy = {}
        j_dummy["true"] = 1 if df[session["tar_col"]][i] else 0
        x = df[session["num_cols"] + session["cat_cols"]].iloc[i].to_frame().T
        j_dummy["pred"] = pipeline.predict(x)[0]
        j.append(j_dummy)
        
    df_pred = pd.DataFrame(j)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, r in df_pred.iterrows():
        if (r["true"] == 1) and (r["pred"] == 1):
            tp = tp + 1
        elif (r["true"] == 1) and (r["pred"] == 0):
            fn = fn + 1
        elif (r["true"] == 0) and (r["pred"] == 1):
            fp = fp + 1
        elif (r["true"] == 0) and (r["pred"] == 0):
            tn = tn + 1

    print("tp: " + str(tp))
    print("tn: " + str(tn))
    print("fp: " + str(fp))
    print("fn: " + str(fn))

def load_nn_pipeline(fp):
    with open(fp, 'rb') as f:
        berka = pickle.load(f)

    return berka["network"]

def transform_input_parameters(df, session,plot_random_performance = False):
    # https://scikit-learn.org/1.0/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay.from_predictions
    col_trans = ColumnTransformer([
        ('num', MinMaxScaler(), session["num_cols"]),
        ('cat', OneHotEncoder(drop='if_binary'), session["cat_cols"])
    ])

    df_transformed = col_trans.fit_transform(df[session["num_cols"] + session["cat_cols"]])
    X = df_transformed[:, :]
    y = df[session["tar_col"]]#.map({False: 0, True: 1})

    # Train test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=10)
    except:
        logger.info("TOO FEW ELEMENTS TO KEEP CLASSES CONSISTENT (stratify)")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


    # rebalance onyl for classification
    # oversampling is important for classification, check difference!!
    # Apply oversampling to the training set
    oversampler = RandomOverSampler(random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)

    metr = {}
    # See the inital model performance
    if plot_random_performance:
        clf = RandomForestClassifier(random_state=10)
        metr["acc_rand"] = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy').mean()
        metr["f1_rand"] = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='f1').mean()
        metr["auc_rand"] = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='roc_auc').mean()

    return col_trans, X_train, X_test, y_train, y_test

def run__randClass_grid(classification_method, X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    params = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
    }
    clf = GridSearchCV(classification_method(random_state=10), param_grid=params, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10), scoring='f1', verbose = 2)
    clf.fit(X_train, y_train)

    #print(clf.best_params_)
    #print(clf.best_score_)

    return clf

def run__randClass_parameter(classification_method, X_train, y_train, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    # n_estimators=200,  max_depth=20, min_samples_split=5, min_samples_leaf=1
    clf = classification_method(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=1, random_state=11)
    clf.fit(X_train, y_train)

    return clf

def get_perf_meas_binary_single(y_true, y_pred, y_true_proba, type):
    perf_meas = {}
    perf_meas[type] = {}
    perf_meas[type]["accuracy"] = accuracy_score(y_true, y_pred)
    perf_meas[type]["f1_score"] = f1_score(y_true, y_pred, average='weighted')
    # we only calculate auc for binary classification.
    try:
        perf_meas[type]["ROC AUC"] = roc_auc_score(y_true, y_true_proba[:, 1])
    except:
        logger.info("ROC AUC NOT AVAILABLE HERE")
        pass
    #perf_meas[type]["confusion_matrix"] = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    perf_meas[type]["y_true"] = y_true
    perf_meas[type]["y_pred"] = y_pred

    return perf_meas

def get_perf_meas_binary(clf, X_train, X_test, y_train, y_test):
    y_train_pred = clf.predict(X_train)
    y_train_proba = clf.predict_proba(X_train)
    y_test_pred = clf.predict(X_test)
    y_test_proba = clf.predict_proba(X_test)
    measure_train = get_perf_meas_binary_single(y_train, y_train_pred, y_train_proba, "train")
    measure_test = get_perf_meas_binary_single(y_test, y_test_pred, y_test_proba, "test")

    return {**measure_train, **measure_test}

def save_pipeline(fn, pipeline, session):
    j_berka = {}
    j_berka["pipeline"] = pipeline
    for k in ["new_cols", "table_connection", "fp_tmp_user_num", "group_expressions", "predict_type", "delimiter", "num_cols", "cat_cols", "tar_col", "map_col_names", "map_datatypes", "col_names_original", "col_names_final", "j__col_dependencies"]:
        j_berka[k] = session[k]
    j_berka["session"] = {k : v for k, v in session.items() if k in ["new_cols", "table_connection", "fp_tmp_user_num", "group_expressions", "predict_type", "delimiter", "num_cols", "cat_cols", "tar_col", "map_col_names", "map_datatypes", "col_names_original", "col_names_final", "j__col_dependencies"]}
    fp = os.path.join("data", "num_networks", f"{fn}.pickle")
    print(session)
    try:
        fp_orig = fp
        with open(fp, 'wb') as f:
            pickle.dump(j_berka, f)
        logger.debug(f"FILE SAVED TO {fp}")
        print(f"FILE SAVED TO {fp}")
        session["fp_pipeline"] = fp
    except:
        fp = os.path.join("data", "num_networks", "dummy.pickle")
        with open(fp, 'wb') as f:
            pickle.dump(j_berka, f)
        logger.debug(f"ERROR: FILE COULDNT BE SAVED TO {fp_orig} SO FILE IS SAVED TO {fp}")
        print(f"ERROR: FILE COULDNT BE SAVED TO {fp_orig} SO FILE IS SAVED TO {fp}")
        session["fp_pipeline"] = fp

def get_name_main_table(m__table_connection, j__df):
    if m__table_connection:
        iter_items = iter(m__table_connection.items())
        name__main_table = next(iter_items)[0][0]
        logger.debug(f"select first entry as main_table: {name__main_table}")
    else:
        name__main_table = next(iter(j__df))
        logger.debug(f"select only entry as main_table: {name__main_table}")
    
    return name__main_table

def join_tables_based_on_connection(j__df, session):
    m__table_connection = t_conn__str2map(session["table_connection"])
    name__main_table = get_name_main_table(m__table_connection, j__df)
    df__joined = join_dfs(j__df, name__main_table, m__table_connection)

    return df__joined

def transform_initial_data(j__df, session):
    # RECALCULATE FROM SESSION
    j__df = adapt_cols_from_session(j__df, session)

    # JOIN
    df__joined = join_tables_based_on_connection(j__df, session)

    # ADD COLUMNS
    df = add_new_cols(session["new_cols"], df__joined.copy(), session)

    return df

def get_df(session, fp = None):
    # so we can specify a different fp
    if not fp:
        fp = session['fp_tmp_user_num']
        print(f"FP FROM SESSION: {fp}")
    else:
        fp = fp
        print(f"FP FROM INPUT: {fp}")
        
    # READ ORIGINAL DATA
        logger.debug(f"READ DATA FROM {fp}")
    j__df = get_j__df(fp, session["delimiter"])
    for t, df in j__df.items():
        session["col_names_original"] = list(df.columns)

    df = transform_initial_data(j__df, session)

    session["col_names_final"] = list(df.columns)

    return df

def get_df_old(session):
    # read data
    j__df = get_j__df(session.get('fp_tmp_user_num'), session.get('delimiter'))

    # map col names and datatypes
    j__df = map__colnames_and_types_df(j__df, session.get("map__col_names"), session.get("map__datatypes"))
    j__df = group_tables(j__df, session.get("group_calculations"), append_tablename = False)

    # get table connections from string
    m__table_connection = t_conn__str2map(session.get("m__table_connection__string"))

    # get main table
    if len(j__df) > 1:
        iter_items = iter(m__table_connection.items())
        name__main_table = next(iter_items)[0][0]
        logger.debug(f"MULTIPLE TABLES: MAIN TABLE IS '{name__main_table}'")
        # join tables
        df__joined = join_dfs(j__df, name__main_table, m__table_connection)
    else:
        name__main_table = next(iter(j__df))
        logger.debug(f"ONLY ONE TABLE: MAIN TABLE IS '{name__main_table}'")
        df__joined = j__df[name__main_table]
    
    # add columns
    new_cols__string = session.get('new_cols__string')
    if new_cols__string:
        df__joined = add_new_cols(new_cols__string, df__joined.copy(), session)

    return df__joined

def trained_pipeline_for_classification(df, session, list__classification_method, param_grid):
    # transform input
    print(df.head())
    col_trans, X_train, X_test, y_train, y_test = transform_input_parameters(df, session)

    # need res per model
    res = {}
    for classification_method in list__classification_method:
        # get classifier
        if session["searchType"]  == "grid":
            clf = run__randClass_grid(classification_method, X_train, y_train, n_estimators = [200], max_depth = [20], min_samples_split = [5], min_samples_leaf = [1])
            clf = run__randClass_parameter(classification_method, X_train, y_train, clf.best_params_["n_estimators"], clf.best_params_["max_depth"], clf.best_params_["min_samples_split"], clf.best_params_["min_samples_leaf"])
        else:
            clf = run__randClass_parameter(classification_method, X_train, y_train, 200, 20, 5, 1)

        # perf meas
        perf_total = get_perf_meas_binary(clf, X_train, X_test, y_train, y_test)

        pipeline = Pipeline([('transformer', col_trans),('classifier', clf)])

        result_visualization = {}
        for t in ["test", "train"]:
            result_visualization[t] = generate_confusion_matrix_plot(confusion_matrix(perf_total[t]["y_true"], perf_total[t]["y_pred"]))

        res[classification_method] = {"pipeline" : pipeline, "metric" : perf_total, "result_visualization" : result_visualization["test"]}

    return res

def trained_pipeline_for_regression(df, session, list__regression_method, param_grid):
    """
    LinearRegression, RandomForestRegressor, MLPRegressor
    """
    # Assuming df is your DataFrame containing the data
    # Split the DataFrame into features (X) and target (y)
    #########################################################################
    #########################DATA############################################
    #########################################################################

    X = df.drop(columns=[session["tar_col"]])  # Features
    y = df[session["tar_col"]]  # Target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #########################################################################
    #########################PIPELINE############################################
    #########################################################################

    # Define numerical and categorical column names
    numerical_cols = session["num_cols"]  # Replace with actual numerical column names
    categorical_cols = session["cat_cols"]  # Replace with actual categorical column names

    # Impute missing values
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Update the preprocessing steps for numerical and categorical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', numerical_imputer),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', categorical_imputer),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps for both numerical and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    #########################################################################
    #########################FIT############################################
    #########################################################################

    res = {}
    for regression_method in list__regression_method:
        # Update the pipeline with the new preprocessing steps
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('regressor', regression_method())])
        # typically bad results for MLP. For good results one could choose
        """
                MLPRegressor(hidden_layer_sizes=(100,),
                             activation='relu',
                             alpha=0.0001,
                             solver='adam',
                             learning_rate='constant',
                             learning_rate_init=0.001,
                             batch_size=32,
                             max_iter=1000,
                             early_stopping=True
                             )
        """
        
        if param_grid:
            print("GRID")

            # Perform grid search to find the best combination of hyperparameters
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)

            # Get the best model from the grid search
            best_model = grid_search.best_estimator_

            # Assuming best_model contains the best model from grid search
            best_regressor = best_model.named_steps['regressor']

            # Define the pipeline with preprocessing and the best regressor
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', best_regressor)])

        pipeline.fit(X_train, y_train)


        #########################################################################
        #########################PERFOMANCE############################################
        #########################################################################
        # Make predictions on the testing data
        predictions = pipeline.predict(X_test)

        # need to write this to the "test" key to be consistent with classification metrics
        metr = {"test": {}}
        metr["test"]["r2"] = r2_score(y_test, predictions)
        metr["test"]["mse"] = mean_squared_error(y_test, predictions)
        metr["test"]["rmse"] = mean_squared_error(y_test, predictions, squared=False)
        metr["test"]["mae"] = mean_absolute_error(y_test, predictions)
        metr["test"]["nmse"] = mean_squared_error(y_test, predictions) / (y_test.max() - y_test.min())
        metr["test"]["y_test"] = y_test
        metr["test"]["predictions"] = predictions

        # create the plot
        plot = generate_result_plot_regression(metr["test"]["y_test"], metr["test"]["predictions"])

        res[regression_method] = {"pipeline" : pipeline, "metric" : metr, "result_visualization" : plot_to_base64(plot), "plot" : plot}

    return res

def set_classification_type(df, session, set_mode = None):
    if not set_mode:
        if has_non_integer_float(df[session["tar_col"]]):
            return "regression"
        else:
            return "classification"

def generate_result_plot_regression(y_test, predictions):
    # Visualize true values vs predicted values
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, predictions, color='blue', label='True vs Predicted')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('True vs Predicted Values')
    ax.legend(loc='best')
    ax.grid(True)
    return fig

def plot_to_base64(plot):
    buf = io.BytesIO()
    plot.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def train_network(df, session, fp_save_pipeline, list__methods, param_grid, called_from_jupyter = False):
    if session["predict_type"] == "regression":
        logger.debug(f"TRAIN REGRESSION NETWORK")
        res = trained_pipeline_for_regression(df, session, list__methods, param_grid)
        res_with_best_metric = min(res.items(), key=lambda x: x[1]["metric"]["test"][session["main_metric"]])
    elif session["predict_type"] == "classification":
        logger.debug(f"TRAIN CLASSIFICATION NETWORK")
        res = trained_pipeline_for_classification(df, session, list__methods, param_grid)
        res_with_best_metric = max(res.items(), key=lambda x: x[1]["metric"]["test"][session["main_metric"]])

    logger.debug([(key.__name__, values.get("metric", {}).get("test", {}).get(session['main_metric'])) for key, values in res.items()])
    logger.debug(f'BEST METRIC based on {session["main_metric"]} ({session["best_type"]}): {res_with_best_metric[0].__name__} with value {res_with_best_metric[1]["metric"]["test"][session["main_metric"]]}')
    pipeline = res_with_best_metric[1]["pipeline"]
    metrics = res_with_best_metric[1]["metric"]
    result_visualization = res_with_best_metric[1]["result_visualization"]

    if fp_save_pipeline:
        save_pipeline(fp_save_pipeline, pipeline, session)
    
    # for debugging purpose
    if called_from_jupyter:
        print("CALLED FROM JUPYTER")
        return pipeline, metrics, result_visualization, session

    return pipeline, metrics, result_visualization, res, res_with_best_metric

def generate_confusion_matrix_plot(confusion_matrix, class_labels=None):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)

    # Add text annotations
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            plt.text(j, i, str(confusion_matrix[i, j]), horizontalalignment='center', verticalalignment='center')

    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Set tick labels dynamically
    num_classes = len(confusion_matrix)
    tick_labels = class_labels if class_labels else [f'Class {i}' for i in range(num_classes)]
    plt.xticks(np.arange(num_classes), tick_labels)
    plt.yticks(np.arange(num_classes), tick_labels)
    
    # Convert plot to bytes
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode as base64 string
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def get_pipeline(session):
    with open(session["fp_pipeline"], 'rb') as f:
        j__info = pickle.load(f)
        
    return j__info["pipeline"]

def pred_regClass_singTa_initialDf(df, session, j__df, pipeline):
    table_name = next(iter(j__df))
    j__df[table_name] = df
    df = transform_initial_data(j__df, session)

    return pipeline.predict(df)

def pred_regClass_singTa_finalDf(df, session, pipeline):
    
    return pipeline.predict(df)

def pred_regClass_singTa_fromSrc(fp, session, pipeline):
    df = get_df(session, fp)
    df = df[session["num_cols"] + session["cat_cols"]]

    return pipeline.predict(df)


