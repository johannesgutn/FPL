from make_data_functions import *
from ml_functions import *
from team_fuctions import *


# This is the main function for the part that gathers data and makes it ready for machine learning
def data_main(last_GW=last_GW):
    # Import latest player data as a json
    get_json('json/fpl_events.json', 'https://fantasy.premierleague.com/api/bootstrap-static/')
    # Open the json file
    with open('json/fpl_events.json') as json_data:
        d = json.load(json_data)
    
    # Get new raw_data
    if os.path.isfile(f'raw_data/raw_data_{season}_{last_GW}.csv'):
        raw_data = pd.read_csv(f'raw_data/raw_data_{season}_{last_GW}.csv',index_col=0)
    else:
        raw_data = json_normalize(d['elements'])
        raw_data.to_csv(f'raw_data/raw_data_{season}_{last_GW}.csv')
        
    # Make everything ready for the new seasono
    if last_GW==0:
        new_season(raw_data)
        raise SystemExit('This is the preseason, so let us stop right here.')
    
    old_data = pd.read_csv(f'player_data/player_data_{season}_{last_GW-1}.csv')
    
    # Get new fixture data
    if os.path.isfile(f'fixture/fixtures_{season}_{last_GW}.csv'):
        fixtures = pd.read_csv(f'fixture/fixtures_{season}_{last_GW}.csv')
    else:
        fixtures = edit_fixture(last_GW)
        fixtures.to_csv(f'fixture/fixtures_{season}_{last_GW}.csv', index = False)

    dgw_teams, bgw_teams= check_DGW_BGW(fixtures)
    
    # Get rid of NaNs and make everything into floats
    raw_data['chance_of_playing_next_round'] = raw_data['chance_of_playing_next_round'].fillna(100)
    raw_data = raw_data.fillna(0)
    raw_data = raw_data.apply(pd.to_numeric, errors='ignore')
    
    # Updating the goal difference
    goal_difference = goal_diff(raw_data,fixtures,last_GW,dgw_teams,bgw_teams)
    goal_difference.to_csv(f'goal_difference/goal_difference_{season}_{last_GW}.csv', index = False)
    
    # Update the new data to include all the additional stats
    new_data = update_new_data(last_GW,raw_data,old_data)
    new_data.to_csv(f'player_data/player_data.csv', index = False) # Save this because we need it to do ML for later GWs
    
    new_data = add_opponent_data(last_GW+1,new_data,goal_difference)
    
    player_type_columns = ['element_type','defender','attacker']
    season_long_columns = ['points_per_game','ict_index_pg','influence_pg','creativity_pg','threat_pg','bps_pg']
    recent_columns = ['form','ep_next','points_pg_last5','ict_index_pg_last5','bps_pg_last5','influence_pg_last5','creativity_pg_last5','threat_pg_last5']
    opponent_columns = ['fixture_difficulty','is_home','opp_GS_pg','opp_GA_pg','opp_GS_last5_pg','opp_GA_last5_pg']
    conceded_columns = []
    for col in ['total_points','ict_index', 'influence', 'creativity', 'threat', 'bps']:
        for lol in ['atk','def','atk_last5','def_last5']:
            conceded_columns.append('opp_'+col+'_conceded_pg_'+lol)
    next_points_columns = ['next_points']
    data_columns = player_type_columns + season_long_columns + recent_columns + opponent_columns + conceded_columns + next_points_columns
    
    # Update the old data with points gotten in the last round to make ML data
    ml_data = make_ml_data(last_GW,new_data,old_data,fixtures,dgw_teams,data_columns)
    
    # Save the machine learning data as csv
    ml_data.to_csv(f'ml_data/ml_data_{season}_{last_GW}.csv', index = False)

    # Add it to the old data to get machine learning data
    if last_GW > 6:
        #total_data_old = pd.read_csv(f'ml_data/total_data_{season}_{last_GW-1}.csv')
        total_data_old = pd.read_csv('ml_data/total_data.csv')
        frames = [ml_data, total_data_old]
        total_data = pd.concat(frames)
        total_data.index = range(len(total_data))
        total_data.to_csv(f'ml_data/total_data_{season}_{last_GW}.csv', index = False)
        total_data.to_csv('ml_data/total_data.csv', index = False)
        if os.path.isfile(f'ml_data/total_data_{season}_{last_GW-2}.csv'):
            os.remove(f'ml_data/total_data_{season}_{last_GW-2}.csv')
    else: 
        total_data_old = pd.read_csv(f'ml_data/total_data.csv')
        total_data_old.to_csv(f'ml_data/total_data_{season}_{last_GW}.csv', index = False)
        if os.path.isfile(f'ml_data/total_data_{season}_{last_GW-2}.csv'):
            os.remove(f'ml_data/total_data_{season}_{last_GW-2}.csv')
        
# This is the function that trains a neural network and does the predictions
def ml_main(last_GW=last_GW):    
    #model = train_neural()
    model = train_model()

    total_data = pd.read_csv('ml_data/total_data.csv')
    goal_difference = pd.read_csv(f'goal_difference/goal_difference_{season}_{last_GW}.csv')
    new_data_bare = pd.read_csv(f'player_data/player_data.csv')
    new_data = add_opponent_data(last_GW,new_data_bare,goal_difference)

    data_columns = list(total_data.columns)

    
    x=5
    predicted_list=[]
    data_columns.remove('next_points')
    for i in range(1,x+1):
        future_data = add_opponent_data(last_GW+i,new_data_bare,goal_difference)
        future_data = predict(model,future_data,data_columns)
        
        future_data_= future_data[['web_name','opponent']+data_columns]
        #future_data_.to_csv(f'player_data/future_data_{season}_{last_GW+i}.csv', index = False)
        
        new_data=insert(new_data,f'predicted_points_{last_GW+i}','id',future_data,'X_points','id')
        new_data=insert(new_data,f'opponent_{last_GW+i}','id',future_data,'opponent','id')
        gd = pd.read_csv(f'goal_difference/goal_difference_{season}_{last_GW}.csv')
        di = gd.set_index("team")["team_name"].to_dict()
        new_data[f'opponent_{last_GW+i}'] = new_data[f'opponent_{last_GW+i}'].map(di)
        predicted_list.append(f'opponent_{last_GW+i}')
        predicted_list.append(f'predicted_points_{last_GW+i}')
        
    new_data['predicted_points_mean'] = round(new_data[predicted_list].sum(axis=1,numeric_only=True)/x,2)
    all_columns = list(new_data.columns)

    column_list = ['web_name']+predicted_list+['predicted_points_mean','points_per_game','form']
    for col in column_list:
        all_columns.remove(col)
    all_columns=column_list+all_columns   
    new_data = new_data[all_columns]
    new_data = new_data.sort_values('predicted_points_mean',ascending=False)
    new_data = new_data.fillna(0)
    
    new_data=new_data.drop_duplicates(subset='id', keep='last')
    

    new_data.to_csv('player_data/player_data.csv', index = False)
    new_data.to_csv(f'player_data/player_data_{season}_{last_GW}.csv', index = False)

##############################################################################################
# This is the main function for making changes to your team
async def team_main(email, password , team_id, last_GW=last_GW):
    
    # Load the ML predictions and reduce the scores of injured players etc
    df_predictions = pd.read_csv(f'player_data/player_data_{season}_{last_GW}.csv')
    df_predictions = reduce_predictions(df_predictions)
    
    team_df, free_transf, bank, wildcard = await get_team(email,password,team_id)
    
    # Make a team object, add ML predictions and divide into bench and XI
    team = Team(team_df, free_transf, bank, wildcard)
    #team.players_per_team = team.add_predictions(df_predictions)
    team.add_predictions(df_predictions)
    team.bench_GK, team.bench_outfield, team.team_XI = team.divide_XI_bench()
    
    # Find the optimal transfer and add it to the team.team_full. Then choose which team to field in 
    team.make_transfers(df_predictions)
    team.team_full.to_csv(f'ml_team/ml_team_{season}_{next_GW}.csv', index = False)
    total_points = team.choose_best_team()
    
    team.update_history(team_id,total_points)
    
