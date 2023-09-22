from configs import *
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse


def print_metric(y_train, y_pred_train,y_test, y_pred_test):
    print(f'Train MAE: {mae(y_train,y_pred_train):.3}, RMSE: {np.sqrt(mse(y_train,y_pred_train)):.3} and r2 score: {r2(y_train,y_pred_train):.3}')
    print(f'Test MAE: {mae(y_test,y_pred_test):.3}, RMSE: {np.sqrt(mse(y_test,y_pred_test)):.3} and r2 score: {r2(y_test,y_pred_test):.3}')

def train_model():
    df = pd.read_csv('ml_data/total_data.csv')

    X = df.drop("next_points", axis=1)
    y = df["next_points"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    params = {'learning_rate': 0.05138716518934617, 'max_depth': 9, 'n_estimators': 115, 'subsample': 0.7937617004685464}
    model = xgb.XGBRegressor(**params, random_state=0)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print_metric(y_train,y_pred_train,y_test,y_pred_test)

    return model

# Take the current data and change the opponent data to match the fixtures in the specific GW
# Then do the predictions based on the model
def predict(model,player_data,data_columns):

    X_predict = player_data[data_columns]
    #X_predict = preprocessing.scale(X_predict)
    
    y_predict = model.predict(X_predict)
    
    player_data['X_points'] = y_predict
    
    # Add expected points for the two DGW matches
    if len(player_data[player_data['id'].duplicated(keep='last')])>0:
        player_data = player_data.sort_values('id', ascending=False)
        dgw1 = player_data[player_data['id'].duplicated(keep='last')]
        dgw2 = player_data[player_data['id'].duplicated(keep='first')]
        player_data = player_data.drop_duplicates(subset='id', keep=False)
        dgw1.set_index('id',inplace = True)
        dgw2.set_index('id',inplace = True)
        dgw1['X_points'] = round(dgw1['X_points'] + dgw2['X_points'],2)
        dgw1.reset_index(inplace=True)
        #player_data = player_data.append(dgw1)
        player_data =pd.concat([player_data,dgw1])
        player_data = player_data.sort_values('X_points', ascending=False)
        player_data.reset_index(inplace=True)

    # Set BGW players' points to 0
    player_data['X_points'] = round(player_data.apply(lambda row: row[['X_points']]*0 if row['fixture_difficulty'] == 0
     else row[['X_points']], axis=1),2)

    return player_data



'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.models import load_model


from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def train_neural():
    #if not os.path.isfile(f'neural_net_{season}_{last_GW-1}.h5'):
    if True:
        # If there is no existing ml model we make it from scratch
        # Load the training data
        df = pd.read_csv('ml_data/total_data.csv')

        df1 = df.drop("next_points", axis=1)
        df2 = df["next_points"].copy()
        X_train, X_test, y_train, y_test = train_test_split(df1, df2, test_size=0.2)
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)
        
        # Now do the fitting
        num_rows, num_cols = X_train.shape
        n_inputs = num_cols
        
        n_inputs = len(X_train[0])

        factor = factor_
        model = Sequential()
        model.add(Dense(n_neurons, input_shape=(n_inputs,), activation='relu'))
        if batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(drop_rate))
        for i in range(n_layers-1):
            model.add(Dense(round(n_neurons*factor), activation='relu'))
            if batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(drop_rate))
            factor = factor * factor
        model.add(Dense(1,))
        model.compile(Adam(learning_rate=learn_rate), 'mean_squared_error')
        #model.compile(optimizer= "adam", loss='mse')

        # Pass several parameters to 'EarlyStopping' function and assigns it to 'earlystopper'
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_, verbose=1, mode='auto')

        # Fits model over 2000 iterations with 'earlystopper' callback, and assigns it to history
        model.fit(X_train, y_train, epochs = 2000, validation_split = 0.2,shuffle = True, verbose = 0, 
                            callbacks = [earlystopper])

        model.save(f'neural_net_{season}_{last_GW}.h5')
    else:
        # If a ml model is already trained we load it and train only on the new data
        df = pd.read_csv(f'ml_data/ml_data_{season}_{last_GW}.csv')
        
        df1 = df.drop("next_points", axis=1)
        df2 = df["next_points"].copy()
        X_train, X_test, y_train, y_test = train_test_split(df1, df2, test_size=0.2)
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)

        model = load_model(f'neural_net_{season}_{last_GW-1}.h5')
        
        earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_, verbose=1, mode='auto')
        model.fit(X_train, y_train, epochs = 2000, validation_split = 0.2,shuffle = True, verbose = 0, 
                            callbacks = [earlystopper])
        model.save(f'neural_net_{season}_{last_GW}.h5')
        if os.path.isfile(f'neural_net_{season}_{last_GW-2}.h5'):
            os.remove(f'neural_net_{season}_{last_GW-2}.h5')



    # Runs model with its current weights on the training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculates and prints rmse score of training and testing data
    print("The RMSE score on the Train set is:\t{:0.3f}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
    print("The RMSE score on the Test set is:\t{:0.3f}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
    
    return model

'''