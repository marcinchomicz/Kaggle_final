import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

import uuid
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
import nni
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization,  Dropout, GlobalMaxPooling1D, MaxPool1D
from tensorflow.keras.layers import LSTM, Concatenate, Conv1D, AveragePooling1D, Bidirectional, Flatten, GRU, Multiply
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, Normalizer, QuantileTransformer
from sklearn.model_selection import GroupShuffleSplit
import inspect

EXPERIMENT_NAME = "AR10_RNN_BASED-v01"
RECORD_MLFLOW=True
MLFLOW_URI = "file://mnt/workdata/_WORK_/Kaggle_202204/mlflow_base_2/"
MLFLOW_COMMENT = '''
    Model based on LSTM - GRU combination
'''
script_path = sys.argv[0]
script_name = os.path.split(script_path)[1].split()[0]

RANDOM_STATE=123
tf.random.set_seed(RANDOM_STATE)

def log_method_definition(method, store_name:str):
    code, line_no = inspect.getsourcelines(method)
    store_name = f"{store_name}_{EXPERIMENT_NAME}.txt"
    mlflow.log_text(''.join(code), store_name)

def prepare_data(params:dict):
    TRPATH = "/mnt/workdata/_WORK_/Kaggle_202204/data_1/"

    train_ts_datafile = os.path.join(TRPATH, "train_transformed_1.pkl")
    train_scalar_datafile = os.path.join(TRPATH, "train_scalarfeats_1.pkl")
    test_ts_datafile = os.path.join(TRPATH, "test_transformed_1.pkl")
    test_scalar_datafile = os.path.join(TRPATH, "test_scalarfeats_1.pkl")
    train_labels = os.path.join(TRPATH, "train_labels.zip")

    train_ts_df = pd.read_pickle(train_ts_datafile)
    train_scalar_df = pd.read_pickle(train_scalar_datafile)
    train_labels = pd.read_pickle(train_labels)
    test_ts_df = pd.read_pickle(test_ts_datafile)
    test_scalar_df = pd.read_pickle(test_scalar_datafile)

    sensor_names = train_ts_df.columns[3:]

    # the data mus be reshapoed to provide 60x13 matrices as input
    SEQ_LEN = 60 # each sequence is 60 samples long
    train_seq_num = int(train_ts_df.shape[0]/SEQ_LEN) # num of train sequences (with repeated subjects)
    test_seq_num = int(test_ts_df.shape[0]/SEQ_LEN) # num of test sequences (with repeated subjects)

    ss = QuantileTransformer(n_quantiles=params['transforming_quantiles'],
                           output_distribution=params['transforming_distrib'],
                           random_state=123)
    ts = QuantileTransformer(n_quantiles=params['transforming_quantiles'],
                             output_distribution=params['transforming_distrib'],
                             random_state=123)

    train_ts_data_=pd.DataFrame(ts.fit_transform(train_ts_df[sensor_names]), columns=sensor_names)
    train_scalar_data_ = pd.DataFrame(ss.fit_transform(train_scalar_df), columns=train_scalar_df.columns)
    test_ts_data_ = pd.DataFrame(ts.transform(test_ts_df[sensor_names]), columns=sensor_names)
    test_scalar_data_ = pd.DataFrame(ss.transform(test_scalar_df), columns=test_scalar_df.columns)

    sensor_num = len(sensor_names)

    train_ts_data = train_ts_data_.values.reshape(train_seq_num, SEQ_LEN, sensor_num)
    test_ts_data = test_ts_data_.values.reshape(test_seq_num, SEQ_LEN, sensor_num)
    train_scalar_data = train_scalar_data_.values
    test_scalar_data = test_scalar_data_.values
    groups=train_ts_df.loc[train_ts_df['step']==0,'subject']

    return train_ts_data, train_scalar_data,  train_labels, test_ts_data, test_scalar_data, groups

def define_model(params: dict):
    input_layer_1 = Input(shape=params['ts_data_dimensions'], name='input_1')

    batch_norm_1 = BatchNormalization(epsilon=1e-6)(input_layer_1)

    lstm0 = Bidirectional(LSTM(units=768, return_sequences=True, dropout= params['lstm_dropout']))(batch_norm_1)
    lstm1 = Bidirectional(LSTM(units=512, return_sequences=True, dropout=params['lstm_dropout']))(lstm0)
    lstm2 = Bidirectional(LSTM(units=384, return_sequences=True, dropout=params['lstm_dropout']))(lstm1)
    lstm3 = Bidirectional(LSTM(units=256, return_sequences=True, dropout=params['lstm_dropout']))(lstm2)
    lstm4 = Bidirectional(LSTM(units=128, return_sequences=True, dropout=params['lstm_dropout']))(lstm3)

    gru1 = Bidirectional(GRU(units=384, return_sequences=True, dropout= params['gru_dropout']))(lstm1)
    matmul1 = Multiply()([lstm2, gru1])

    gru2 = Bidirectional(GRU(units=256, return_sequences=True, dropout= params['gru_dropout']))(matmul1)
    matmul2 = Multiply()([lstm3, gru2])

    gru3 = Bidirectional(GRU(units=128, return_sequences=True, dropout= params['gru_dropout']))(matmul2)

    concat = Concatenate(axis=2)([lstm0, lstm2, lstm4, lstm3, gru1, gru2, gru3])
    gmp = GlobalMaxPooling1D()(concat)
    drop = Dropout(rate=params['dense_0_drop_ratio'])(gmp)
    dense = Dense(params["dense_0_units"],activation='relu')(drop)
    drop = Dropout(rate=params['dense_1_drop_ratio'])(dense)
    dense = Dense(params["dense_0_units"], activation='relu')(drop)
    output_layer = Dense(1, activation="sigmoid", name='output_layer')(dense)
    model = Model(inputs=input_layer_1,
                  outputs=output_layer,
                  name=script_name + '-model')
    return model

def epoch_reporter(epoch, logs):
    '''@nni.report_intermediate_result(logs['val_loss'])'''
    a=logs
    if RECORD_MLFLOW:
        mlflow.log_metrics(logs)

def define_params():
    params = {
        'ts_data_dimensions': (60,  273),
        'n_splits': 10,
    }

    '''@nni.variable(nni.quniform(100, 5000, 100), name=transforming_quantiles)'''
    transforming_quantiles = 1000
    params['transforming_quantiles'] = int(transforming_quantiles)

    '''@nni.variable(nni.choice('normal', 'uniform'), name=transforming_distrib)'''
    transforming_distrib = 'normal'
    params['transforming_distrib'] = transforming_distrib

    '''@nni.variable(nni.quniform(5e-5, 1e-3, 5e-5), name=initial_lr)'''
    initial_lr =3e-4
    params['initial_lr'] = initial_lr

    '''@nni.variable(nni.quniform(0.75, 0.98, 1e-3), name=adam_beta1)'''
    adam_beta1=0.9
    params['adam_beta1']=adam_beta1

    '''@nni.variable(nni.choice(0,1), name=adam_amsgrad)'''
    adam_amsgrad=0
    params['adam_amsgrad']=True if adam_amsgrad==1 else False

    '''@nni.variable(nni.quniform(32, 128, 16), name=batch_size)'''
    batch_size = 32
    params['batch_size'] = int(batch_size)

    '''@nni.variable(nni.uniform(0, 0.2), name=lstm_dropout)'''
    lstm_dropout = 0.05
    params['lstm_dropout'] = lstm_dropout

    '''@nni.variable(nni.uniform(0, 0.2), name=gru_dropout)'''
    gru_dropout = 0.05
    params['gru_dropout'] = gru_dropout

    '''@nni.variable(nni.quniform(0.0, 0.5, 1e-3), name=dense_0_drop_ratio)'''
    dense_0_drop_ratio = 0.1
    params["dense_0_drop_ratio"] = dense_0_drop_ratio

    '''@nni.variable(nni.quniform(0.0, 0.5, 1e-3), name=dense_1_drop_ratio)'''
    dense_1_drop_ratio = 0.1
    params["dense_1_drop_ratio"] = dense_1_drop_ratio

    '''@nni.variable(nni.quniform(256, 2048, 128), name=dense_0_units)'''
    dense_0_units = 1024
    params["dense_0_units"] = int(dense_0_units)

    '''@nni.variable(nni.quniform(64, 384, 64), name=dense_1_units)'''
    dense_1_units = 256
    params["dense_1_units"] = int(dense_1_units)
    return params

if RECORD_MLFLOW:
    mlflow.set_tracking_uri(uri=MLFLOW_URI)
    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    mlflow.start_run(run_name=nni.get_trial_id() if nni.get_trial_id()!= "STANDALONE" else str(uuid.uuid1()).split("-")[0].upper())
    mlflow.set_tag('script', script_path)
    mlflow.autolog(log_models=False)
    log_method_definition(prepare_data, 'data_preparation_method')
    log_method_definition(define_model, 'model_definition_method')
    log_method_definition(define_params, 'param_definition_method')


MAX_EPOCH = 100
session_params = define_params()
model = define_model(session_params)

if RECORD_MLFLOW:
    mlflow.log_params(params=session_params)

    # log params as dictionary
    s=''
    for k in session_params.keys():
        s=s+'\n'+k+':'+str(session_params[k])+','
    mlflow.log_text(s, 'model_params.txt')

train_ts_data, train_scalar_data, train_labels, test_ts_data, test_scalar_data, groups = prepare_data(session_params)
splitter = GroupShuffleSplit(n_splits=session_params['n_splits'], random_state=123)
splits=splitter.split(X=train_ts_data, y=train_labels[["state"]].values, groups=groups)

idx = 0
global_results = dict()
RUN_SINGLE=True

for split in splits:
    train_ts_X = train_ts_data[split[0]]
    train_scalar_X = train_scalar_data[split[0]]
    train_y = train_labels[["state"]].values[split[0]]
    val_ts_X = train_ts_data[split[1]]
    val_scalar_X = train_scalar_data[split[1]]
    val_y = train_labels[["state"]].values[split[1]]

    print(f"Split number: {idx}, Train: {train_ts_X.shape}, validation: {val_ts_X.shape}")
    # strategy =  tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = define_model(session_params)
    model.compile(
        optimizer=Adam(
            learning_rate=session_params['initial_lr'],
            beta_1 = session_params['adam_beta1'],
            amsgrad = session_params['adam_amsgrad']),
        loss='binary_crossentropy',
        metrics=[
            AUC(name='AUC'),
        ])
    # train
    history = model.fit(
        x=train_ts_X,
        y=train_y,
        callbacks=[
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=3,
                              verbose=1),
            EarlyStopping(monitor='val_loss', patience=6, verbose=1),
            EarlyStopping(monitor='val_AUC', patience=5, baseline=0.8, mode='max', verbose=1),
            LambdaCallback(on_epoch_end=epoch_reporter)
        ],
        batch_size=session_params['batch_size'],
        epochs=MAX_EPOCH, steps_per_epoch = (train_ts_X.shape[0] // session_params['batch_size']) // 4,
        validation_data=(val_ts_X, val_y)
    )
    # store model diagram
    if RECORD_MLFLOW:
        diagram_file = f'diagram_{script_name}_{nni.get_trial_id()}.png'
        tf.keras.utils.plot_model(model,to_file=diagram_file,
                                  show_shapes=True,show_dtype=True, show_layer_names=True)
        mlflow.log_artifact(diagram_file)
        os.remove(path=diagram_file)
    else:
        tf.keras.utils.plot_model(model,to_file=f'diagram_{script_name}.png',
                                  show_shapes=True,show_dtype=True, show_layer_names=True)

    if RUN_SINGLE:
        result_df = pd.DataFrame.from_dict(history.history)
        results = result_df.loc[result_df['val_loss'] == result_df['val_loss'].min()]
        results = results.to_dict(orient='list')
        results_nni = {'default': results['val_loss'][0],
                       'loss': results['loss'][0],
                       'val_AUC': results['val_AUC'][0],
                       'AUC': results['AUC'][0]}
        '''@nni.report_final_result(results_nni)'''
        if RECORD_MLFLOW:
            results_={'fin_val_loss': results['val_loss'][0],
                       'fin_loss': results['loss'][0],
                       'fin_val_AUC': results['val_AUC'][0],
                       'fin_AUC': results['AUC'][0]}
            mlflow.log_metrics(results_)
        break
    else:
        idx += 1

_=pd.DataFrame.from_dict(history.history).reset_index()
fig, ax=plt.subplots(1,2, figsize=(18,8))
sns.lineplot(x='index', y='loss', data=_, ax=ax[0], linewidth=3, color='blue')
sns.lineplot(x='index', y='val_loss', data=_, ax=ax[0], linewidth=3, color='red')
ax[0].grid('both')
ax[0].set_title('Training and validation loss')
sns.lineplot(x='index', y='AUC', data=_, ax=ax[1], linewidth=3, color='blue')
sns.lineplot(x='index', y='val_AUC', data=_, ax=ax[1], linewidth=3, color='red')
ax[1].grid('both')
ax[1].set_title('Training and validateion AUC')
if RECORD_MLFLOW:
    mlflow.log_figure(fig, 'training_results.png')

plot_model(model, show_shapes=True, show_layer_names=True, show_dtype=True,
           to_file='model_diagram.png')
if RECORD_MLFLOW:
    mlflow.log_artifact(local_path='model_diagram.png')
    mlflow.log_artifact(local_path = script_path)