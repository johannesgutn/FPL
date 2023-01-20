from configs import *



def edit_fixture(GW):
    get_json(f'json/fpl_fix_{season}_{GW}.json', 'https://fantasy.premierleague.com/api/fixtures')
    with open(f'json/fpl_fix_{season}_{GW}.json') as json_data:
        d = json.load(json_data)

    # Making dataframe
    df = json_normalize(d)

    # Editing dataframe to contain the columns we want and the GW we want
    df1 = df[df.event == GW]
    df_a = df1[['event','team_a', 'team_a_difficulty','team_a_score','team_h_score','team_h']]
    df_h = df1[['event','team_h', 'team_h_difficulty','team_h_score','team_a_score','team_a']]
    df_a['is_home']=0
    df_h['is_home']=1
    df_a.columns = ['event','team', 'fixture_difficulty','GS','GA','opponent','is_home']
    df_h.columns = ['event','team', 'fixture_difficulty','GS','GA','opponent','is_home']
    frames = [df_a,df_h]
    df1 = pd.concat(frames)
    df1 = df1.sort_values('team', ascending=True)
    df1.index = range(len(df1))

    return df1



def check_DGW_BGW(fixture_df):
    teams = list(fixture_df['team'])
    dgwt = list(set([x for x in teams if teams.count(x) > 1]))
    bgwt = []
    for i in range(1, 21):
        if i not in teams:
            bgwt.append(i)
            
    DGW_teams = np.array(dgwt)
    BGW_teams = np.array(bgwt)
    return DGW_teams, BGW_teams



#fixtures is fixtures
def goal_diff(df,fixtures,GW,DGW_teams,BGW_teams):
    #Opens old fixture stuff
    goal_diff = pd.read_csv(f'goal_difference/goal_difference_{season}_{GW-1}.csv')

    #DGW stuff.
    if len(DGW_teams)>0:
        print('Last: DGW')
        for i in DGW_teams:
            fixtures_temp=fixtures[fixtures.team==i]
            fixtures_temp.index = range(len(fixtures_temp))
            #fixtures_temp.loc[1,['GS']] = fixtures_temp.loc[0,['GS']] + fixtures_temp.loc[1,['GS']]
            #fixtures_temp.loc[1,['GA']] = fixtures_temp.loc[0,['GA']] + fixtures_temp.loc[1,['GA']]
            fixtures_temp.loc[0,['GS']] = fixtures_temp['GS'].sum()
            fixtures_temp.loc[0,['GA']] = fixtures_temp['GA'].sum()
            fixtures_temp=fixtures_temp.drop_duplicates(subset=['team'], keep='first')
            frames=[fixtures,fixtures_temp]
            fixtures=pd.concat(frames)
        fixtures=fixtures.drop_duplicates(subset=['team'], keep='last')
        fixtures = fixtures.sort_values('team', ascending=True)
        fixtures.index = range(len(fixtures))

    # BGW: trick to make us able to update goal difference for the teams that played
    if len(BGW_teams)>0:
        print('Last: BGW')
        fixtures.set_index('team',inplace = True)
        fixtures = fixtures.reindex(range(1,21))
        fixtures = fixtures.fillna(0)
        fixtures.reset_index(inplace=True)

    # Updating goal difference.
    goal_diff['GS_tot']=goal_diff['GS_tot'] + fixtures['GS']
    goal_diff['GA_tot']=goal_diff['GA_tot'] + fixtures['GA']

    # DGW/BGW: Updating games played, and adding/subtracting matches for DGW/BGW teams
    teams = list(fixtures['team'])
    for team in range(1,21):
        goal_diff.loc[team-1,['games_played']]=goal_diff.loc[team-1,['games_played']]+teams.count(team)
        
    '''for index, row in goal_diff.iterrows():
        row['games_played']=row['games_played']+teams.count(row['team'])
        print(row['team'])
        print(teams.count(row['team']))
    goal_diff['games_played']=goal_diff['games_played'] + 1
    if len(DGW_teams)>0:
        goal_diff.loc[DGW_teams-1,'games_played'] = goal_diff.loc[DGW_teams-1,'games_played'] + 1
    if len(BGW_teams)>0:
        goal_diff.loc[BGW_teams-1,'games_played'] = goal_diff.loc[BGW_teams-1,'games_played'] - 1
    '''

    # Per game
    goal_diff['GS_pg']=round(goal_diff['GS_tot']/goal_diff['games_played'],2)
    goal_diff['GA_pg']=round(goal_diff['GA_tot']/goal_diff['games_played'],2)

    # Getting the goal diff for the last 5 GWs
    if GW-5 >=0:
        goal_diff5 = pd.read_csv(f'goal_difference/goal_difference_{season}_{GW-5}.csv')
    else:
        goal_diff5 = pd.read_csv(f'goal_difference/goal_difference_{season}_0.csv')
    goal_diff['GS_last5'] = goal_diff['GS_tot'] - goal_diff5['GS_tot']
    goal_diff['GA_last5'] = goal_diff['GA_tot'] - goal_diff5['GA_tot']

    # Because of DGW/BGW we need to know how many matches they played in the last 5 GWs. Then gd per game.
    goal_diff['games_played_last5'] = goal_diff['games_played'] - goal_diff5['games_played']
    goal_diff['GS_last5_pg']=round(goal_diff['GS_last5']/goal_diff['games_played_last5'], 2)
    goal_diff['GA_last5_pg']=round(goal_diff['GA_last5']/goal_diff['games_played_last5'], 2)
    
    
    # conceded points and other things
    # conceded_atk means points scored by opponent attackers and thus reflects on the defence of the team
    df = insert(df,'last_opponent','team',fixtures,'opponent','team')
    df_prev = pd.read_csv(f'player_data/player_data_{season}_{GW-1}.csv')
    
    df=df.drop_duplicates(subset='id', keep='last')
    df_prev=df_prev.drop_duplicates(subset='id', keep='last')
    
    df.index=df.id
    df_prev.index=df_prev.id
    
    df = df.fillna(0)
    
    for col in ['total_points','ict_index', 'influence', 'creativity', 'threat', 'bps']:
        df[f'{col}_last_game']=df[col]-df_prev[col]
        
        for i in range(1,21):
            # Conceded points
            df_team=df[df.last_opponent==i]
            
            #conceded = df_team[f'{col}_last_game'].sum()
            #goal_diff.loc[i-1,f'{col}_conceded']=goal_diff.loc[i-1,f'{col}_conceded']+conceded

            df_team_atk = df_team[df_team.element_type >2.5]
            conceded = df_team_atk[f'{col}_last_game'].sum()
            goal_diff.loc[i-1,f'{col}_conceded_atk']=goal_diff.loc[i-1,f'{col}_conceded_atk']+conceded

            df_team_def = df_team[df_team.element_type < 2.5]
            conceded = df_team_def[f'{col}_last_game'].sum()
            goal_diff.loc[i-1,f'{col}_conceded_def']=goal_diff.loc[i-1,f'{col}_conceded_def']+conceded
            
            # Scored points
            df_team = df[df.team==i]
            
            df_team_atk = df_team[df_team.element_type >2.5]
            goal_diff.loc[i-1,f'{col}_scored_atk']=df_team_atk[f'{col}'].sum()
            
            df_team_def = df_team[df_team.element_type < 2.5]
            goal_diff.loc[i-1,f'{col}_scored_def']=df_team_def[f'{col}'].sum()

        #goal_diff[f'{col}_conceded_per_game']=round(goal_diff[f'{col}_conceded']/goal_diff['games_played'],2)
        goal_diff[f'{col}_conceded_pg_atk']=round(goal_diff[f'{col}_conceded_atk']/goal_diff['games_played'],2)
        goal_diff[f'{col}_conceded_pg_def']=round(goal_diff[f'{col}_conceded_def']/goal_diff['games_played'],2)
        goal_diff[f'{col}_conceded_pg_atk_last5']=round((goal_diff[f'{col}_conceded_atk']-goal_diff5[f'{col}_conceded_atk'])/goal_diff['games_played_last5'],2)
        goal_diff[f'{col}_conceded_pg_def_last5']=round((goal_diff[f'{col}_conceded_def']-goal_diff5[f'{col}_conceded_def'])/goal_diff['games_played_last5'],2)

        goal_diff[f'{col}_scored_pg_atk']=round(goal_diff[f'{col}_scored_atk']/goal_diff['games_played'],2)
        goal_diff[f'{col}_scored_pg_def']=round(goal_diff[f'{col}_scored_def']/goal_diff['games_played'],2)
        #goal_diff[f'{col}_scored_pg_atk_last5']=round((goal_diff[f'{col}_scored_atk']-goal_diff5[f'{col}_scored_atk'])/goal_diff['games_played_last5'],2)
        #goal_diff[f'{col}_scored_pg_def_last5']=round((goal_diff[f'{col}_scored_def']-goal_diff5[f'{col}_scored_def'])/goal_diff['games_played_last5'],2)
        

    goal_diff=round(goal_diff,1)

    return goal_diff


# This is not made to incorporate DGW and BGW
def update_nailed(new_data,old1_data,old5_data):
    
    # First update season long stats
    new_data['minutes_last_game'] = new_data['minutes'] - old1_data['minutes']
    new_data['games_injured']=old1_data['games_injured']
    new_data['games_dropped']=old1_data['games_dropped']
    new_data['games_played']=old1_data['games_played']
    new_data['games_started']=old1_data['games_started']
    new_data['games_started']=new_data.apply(lambda row: row[['games_started']]+1 if row['minutes_last_game'] > 45
                                else row[['games_started']], axis=1)
    new_data['games_played']=new_data.apply(lambda row: row[['games_played']]+1 if row['minutes_last_game'] > 0.01
                                else row[['games_played']], axis=1)
    new_data['games_played']=new_data.apply(lambda row: row[['games_played']]+1 if row['minutes_last_game'] > 90.01
                                else row[['games_played']], axis=1)
    new_data['games_dropped']=new_data.apply(lambda row: row[['games_dropped']]+1 if row['minutes_last_game'] <= 45
                               and row['status']== 'a'
                                else row[['games_dropped']], axis=1)
    new_data['games_injured']=new_data.apply(lambda row: row[['games_injured']]+1 if row['minutes_last_game'] <= 45
                               and row['status']!= 'a'
                                else row[['games_injured']], axis=1)
    new_data[['minutes_last_game','games_injured','games_dropped','games_played','games_started']] = new_data[['minutes_last_game','games_injured','games_dropped','games_played','games_started']].fillna(0)
    new_data['nail']=round(new_data['games_started']/(new_data['games_started']+new_data['games_dropped']),2)

    # Update stats from 5 GWs ago
    old5_data = old5_data.fillna(0)
    new_data['games_played_last5']=round(new_data.games_played - old5_data.games_played)

    new_data['minutes_last5'] = new_data.minutes - old5_data.minutes
    new_data['minutes_pg_last5'] = round(new_data.minutes_last5/new_data.games_played_last5)
    
    for col in ['games_injured','games_dropped','games_played','games_started']:
        new_data[col+'_last5']=new_data[col]-old5_data[col]
    new_data['nail_last5']=round(new_data['games_started_last5']/(new_data['games_started_last5']+new_data['games_dropped_last5']),2)
    new_data['nail_last5']=new_data['nail_last5'].fillna(0)
    
    
    return new_data



def update_new_data(GW,new_data,old1_data):
    
    # Find out points, ict, influence, creativity, threat and bps per game in the last 5 games. Start by loading 5 GWs old data
    if GW-5 >= 0:
        old5_data = pd.read_csv(f'player_data/player_data_{season}_{GW-5}.csv')
    else:
        old5_data = pd.read_csv(f'player_data/player_data_{season}_0.csv')
    
    # Fix if last GW was DGW
    old1_data = old1_data.drop_duplicates(subset='id', keep='last')
    old5_data = old5_data.drop_duplicates(subset='id', keep='last')
    
    new_data.set_index('id',inplace = True)
    old1_data.set_index('id',inplace = True)
    old5_data.set_index('id',inplace = True)
    
    # One-hot encode atk or def
    new_data['defender'] = 0
    new_data['attacker'] = 0
    new_data['defender']=new_data.apply(lambda row: row[['defender']]+1 if row['element_type'] < 2.5
                                else row[['defender']], axis=1)
    new_data['attacker']=new_data.apply(lambda row: row[['attacker']]+1 if row['element_type'] > 2.5
                                else row[['attacker']], axis=1)
    
    new_data = update_nailed(new_data,old1_data,old5_data)

    # Updating season long stats
    for col in ['ict_index','bps','influence','creativity','threat']:
        new_data[col+'_pg']=round(new_data[col]/new_data.games_played,2)

    # Updating stats last 5 GWs
    new_data['points_pg_last5']=round((new_data['total_points']-old5_data['total_points'])/new_data['games_played_last5'],2)
    for col in ['ict_index','bps','influence','creativity','threat']:
        new_data[col+'_pg_last5']=round((new_data[col]-old5_data[col])/new_data['games_played_last5'],2)
        
    new_data = new_data.fillna(0)
    new_data.reset_index(inplace = True)
    
    return new_data



def add_opponent_data(GW,new_data,gd):
    # Making fixture for next GW and check if its DGW/BGW
    fix = edit_fixture(GW)

    DGW_teams, BGW_teams= check_DGW_BGW(fix)

    # Duplicate players with DGW so they have both games
    if len(DGW_teams) > 0:
        # Make new_data of players with DGW
        new_data_dgw1 = pd.DataFrame()
        for team in DGW_teams:
            new_data_dgw1=new_data_dgw1.append(new_data[new_data.team == team])
            new_data = new_data[new_data.team != team]
        new_data_dgw2 = new_data_dgw1.copy()
        fix_dgw = pd.DataFrame()
        # Get the double fixtures
        for team in DGW_teams:
            fix_dgw=fix_dgw.append(fix[fix.team == team])
        fix_dgw1=fix_dgw.iloc[::2]
        fix_dgw2=fix_dgw.iloc[1::2]
        # Set in fixture diff, opponent and is_home and merge them all together in the end
        for col in ['fixture_difficulty','opponent','is_home']:
            new_data = insert(new_data,col,'team',fix,col,'team')
            new_data_dgw1 = insert(new_data_dgw1,col,'team',fix_dgw1,col,'team')
            new_data_dgw2 = insert(new_data_dgw2,col,'team',fix_dgw2,col,'team')
            
        frames = [new_data,new_data_dgw1,new_data_dgw2]
        new_data = pd.concat(frames)
        new_data = new_data.sort_values('team',ascending=True)

    # Insert fixture difficulty stuff and opponent
    if len(DGW_teams) == 0:
        for col in ['fixture_difficulty','opponent','is_home']:
            new_data = insert(new_data,col,'team',fix,col,'team')
    
    # Insert goal difference stuff and conceded stuff
    goal_difference_cols = ['GS_pg','GA_pg','GS_last5_pg','GA_last5_pg']
    
    conceded_cols = []
    for lol in ['atk','def','atk_last5','def_last5']:
        for col in ['total_points','ict_index', 'influence', 'creativity', 'threat', 'bps']:
            conceded_cols.append(col+'_conceded_pg_'+lol)
            
    cols = goal_difference_cols + conceded_cols
    for col in cols:
        new_data = insert(new_data,'opp_'+col,'opponent',gd,col,'team')

    # Just to make sure there are no infinities or NaN
    new_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    return new_data



# new_player_data is new data, ml_data is old data
def make_ml_data(GW,new_player_data,ml_data,fix,DGW_teams,data_columns):

    # Taking the newest points and minutes and adding it to the old dataframe
    ml_data = insert(ml_data,'next_points','id',new_player_data,'event_points','id')
    ml_data = insert(ml_data,'next_minutes','id',new_player_data,'minutes','id')

    # Take out those who played less than 30 minutes last match
    ml_data['minutes_diff']=ml_data.next_minutes - ml_data.minutes
    ml_data = ml_data[ml_data.minutes_diff > 30]
    
    # Also take out those with low form
    ml_data = ml_data[ml_data.form > 0.1]

    # DGW: delete DGW teams
    if len(DGW_teams)>0:
        for i in DGW_teams:
            ml_data = ml_data[ml_data.team != i]

    # Keep only the columns we want for machine learning
    ml_data = ml_data[data_columns]

    # Before BGW: Take out those that have NaN in fixture_difficulty
    ml_data = ml_data[ml_data.fixture_difficulty > 0.1]
    
    # Drop entries that for some reason have 0 in all last5 colums
    ind=ml_data[(ml_data["points_pg_last5"] == 0) & (ml_data["ict_index_pg_last5"] == 0)&(ml_data["bps_pg_last5"] == 0)&
               (ml_data["influence_pg_last5"] == 0)& (ml_data["creativity_pg_last5"] == 0)& (ml_data["threat_pg_last5"] == 0)].index
    ml_data=ml_data.drop(ind)
    ml_data.index = range(len(ml_data))
    
    return ml_data


def new_season(df):
    # This long shit makes clean slate for new season player data df
    #######
    df_last = pd.read_csv(f'player_data/player_data_{last_season}_38.csv')
    cols_all = list(df_last.columns)

    cols_raw = list(df.columns)
    
    cols_new = cols_all.copy()
    for col in cols_raw:
        if col in cols_new:
            cols_new.remove(col)
    
    #or col in cols_new:
    #    df[col]=0
        
    
    # Puttin in old data that is not there in raw
    for col in cols_new:
        insert(df,col,'code',df_last,col,'code')
    
    df['chance_of_playing_next_round'] = df['chance_of_playing_next_round'].fillna(100)
    df = df.fillna(0)
    
    df.to_csv(f'player_data/player_data_{season}_preseason.csv', index = False)
    
    cols_keep = ['code','id','chance_of_playing_next_round','element_type','web_name','team']
    for col in cols_keep:
        cols_all.remove(col)
        
    for col in cols_all:
        df[col]=0
            
    df['predicted_points_1']=0
    df['predicted_points_mean']=0
    
    df.to_csv(f'player_data/player_data_{season}_0.csv', index = False)
    #######
    
    # Fix goal diff file with new team names
    df = pd.read_csv(f'goal_difference/goal_difference_{last_season}_0.csv')
    with open('json/fpl_events.json') as json_data:
        d = json.load(json_data)
    teams=json_normalize(d['teams'])
    
    insert(df, 'team_name', 'team', teams, 'short_name', 'id')
    
    
    
    df.to_csv(f'goal_difference/goal_difference_{season}_0.csv', index = False)
    




def train_neural():
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
    model.compile(Adam(lr=learn_rate), 'mean_squared_error')

    # Pass several parameters to 'EarlyStopping' function and assigns it to 'earlystopper'
    earlystopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience_, verbose=1, mode='auto')

    # Fits model over 2000 iterations with 'earlystopper' callback, and assigns it to history
    history = model.fit(X_train, y_train, epochs = 2000, validation_split = 0.2,shuffle = True, verbose = 0, 
                        callbacks = [earlystopper])
    
    # Runs model with its current weights on the training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculates and prints rmse score of training and testing data
    print("The RMSE score on the Train set is:\t{:0.3f}".format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
    print("The RMSE score on the Test set is:\t{:0.3f}".format(np.sqrt(mean_squared_error(y_test, y_test_pred))))
    
    return model


# Take the current data and change the opponent data to match the fixtures in the specific GW
# Then do the predictions based on the model
def predict(model,player_data,data_columns):

    X_predict = player_data[data_columns]
    X_predict = preprocessing.scale(X_predict)
    
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
        player_data = player_data.append(dgw1)
        player_data = player_data.sort_values('X_points', ascending=False)
        player_data.reset_index(inplace=True)

    # Set BGW players' points to 0
    player_data['X_points'] = round(player_data.apply(lambda row: row[['X_points']]*0 if row['fixture_difficulty'] == 0
     else row[['X_points']], axis=1),2)

    return player_data