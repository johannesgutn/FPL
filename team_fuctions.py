from configs import *

# These are from the team stuff

def reduce_predictions(pred):
    for i in f'{next_GW}','mean':
        pred['predicted_points_'+i+'_raw']=pred['predicted_points_'+i]
        pred['predicted_points_'+i] = round(pred['predicted_points_'+i]*pred['chance_of_playing_next_round']/100,2)
        pred['predicted_points_'+i] = round(pred.apply(lambda row: row[['predicted_points_'+i]]*row['nail']/0.85 if row['nail'] < 0.8 and row['nail_last5'] < 0.8
         else row[['predicted_points_'+i]], axis=1),2)
        pred['predicted_points_'+i]=pred.apply(lambda row: row[['predicted_points_'+i]]*row['form'] if row['form'] < 1 
         else row[['predicted_points_'+i]], axis=1)
        if next_GW > 6:
            pred['predicted_points_'+i]=pred.apply(lambda row: row[['predicted_points_'+i]]*row['games_played_last5']/3 if row['games_played_last5'] < 3 
             else row[['predicted_points_'+i]], axis=1)
            pred['predicted_points_'+i]= round(pred.apply(lambda row: row[['predicted_points_'+i]]*row['minutes_pg_last5']/20 if row['minutes_pg_last5'] < 20 
             else row[['predicted_points_'+i]], axis=1),2)
    
    pred = pred.sort_values('predicted_points_mean', ascending=False)
    
    return pred



async def get_team(email,password,team_id):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email, password)
        user = await fpl.get_user(team_id)
        team_ = await user.get_team()
        transfer_status_ = await user.get_transfers_status()
        
        team = json_normalize(team_)
        team['id']=team['element']
        
        transfer_status = json_normalize(transfer_status_)
        
        wildcard = False
        free_transf = transfer_status.loc[0,'limit']
        if str(free_transf)=='None':
            wildcard = True
        else:
            free_transf = int(free_transf)

        bank = float(transfer_status.loc[0,'bank'])
                
    return team, free_transf, bank, wildcard



class Team:
    def __init__(self,team_df, free_transf, bank, wildcard):
        self.team_full = team_df
        self.free_transf = free_transf
        self.bank = bank
        self.wildcard = wildcard
        
    
    def add_predictions(self,df_predictions):
        columns_to_add = ['web_name','team','chance_of_playing_next_round', 'element_type','event_points', 
                          'total_points','form','points_per_game',f'predicted_points_{next_GW}',
                          'predicted_points_mean',f'predicted_points_{next_GW}_raw','predicted_points_mean_raw']
        for col in columns_to_add:
            insert(self.team_full,col,'id',df_predictions,col,'id')
            
        # Make list of teams we have 3 players from
        players_per_team = dict(self.team_full['team'].value_counts())
        for i in range(1,21):
            if i not in players_per_team.keys():
                players_per_team[i]=0
        
        self.players_per_team = players_per_team
        
        #return players_per_team
    
    def divide_XI_bench(self):
        # Make dataframe with 2nd GK
        team=self.team_full.copy()

        team = team.sort_values('selling_price', ascending=True)
        
        GKs = team[team.element_type == 1]
        GKs.index = range(len(GKs))
        if GKs.loc[0,'selling_price'] == GKs.loc[0,'selling_price']:
            GKs = GKs.sort_values('predicted_points_mean', ascending=True)
            GK2=GKs.head(1) 
        else:
            GK2=GKs.head(1)
        
        
        team=team[team['id'] != int(GK2['id'])]
        
        pos_price = {2:45, 3:50, 4:50}
        full_bench = GK2
        for pos in pos_price:  
            posses = team[team.element_type == pos]
            posses_bench = posses[posses.selling_price < pos_price[pos]]
            #full_bench = full_bench.append(posses_bench)
            full_bench = pd.concat([full_bench,posses_bench])
        bench=full_bench[full_bench['id'] != int(GK2['id'])]
        if len(bench) > 3:
            bench = bench.sort_values('total_points', ascending=True)
            bench = bench.head(3)
        if len(bench) < 3:
            bench = team.head(3)
        team = team.sort_values('selling_price', ascending=True)
        for index, player in bench.iterrows():
            team = team[team.id != player['id']]
        team = team.sort_values('position', ascending=True)
        
        return GK2, bench, team
    
    def make_transfers(self,df_predictions):
        # Now we need to figure out which action to take. Zero, one or two changes. Change bench or outfield player
                
        # Find cheapest and most expensive player
        low_price = df_predictions['now_cost'].min()
        high_price = df_predictions['now_cost'].max()

        # Take out the players we already have
        players = list(self.team_full['id'])
        id_name = df_predictions.set_index('id')['web_name'].to_dict()
        for player in players:
            df_predictions = df_predictions[df_predictions.id != player]
        df_predictions.index = range(len(df_predictions))
        
        # How many outfield players are bad:
        team = self.team_XI.copy()
        team = team.sort_values('predicted_points_mean', ascending=True)
        worst_XI = team[team.predicted_points_mean < 3]
        worst_XI_nr = len(worst_XI)
        
        injured_XI_nr = len(team[team.predicted_points_mean < 1])

        # How many bench players are bad:
        bench = self.bench_outfield.copy()
        bench = bench.sort_values('predicted_points_mean', ascending=True)
        worst_bench = bench[bench.predicted_points_mean < 1]
        worst_bench_nr = len(worst_bench)
        
        free_transf = self.free_transf
        # This logic decides what to do
        # It returns a df with columns 'player','points','replacement','rep_points','points_gained'
        if last_GW == 1 and injured_XI_nr == 0:
            maketransf = False
            print('Save a transfer')
            transfer = pd.DataFrame(columns=['player','points','replacement','rep_points','points_gained'], dtype=object)
        else:
            maketransf = True
                        
        worst_XI_nr=1
        if maketransf:
            if free_transf == 2 and worst_XI_nr >= 2:
                print("2 normal transfer")
                transfer = self.two_transf(df_predictions,low_price,high_price)
            elif free_transf == 2 and worst_XI_nr == 1 and worst_bench_nr > 2:
                print("1 norm transf 1 bench transfer")
                transfer = self.one_transf1bench(df_predictions)
            elif worst_XI_nr == 0 and worst_bench_nr > 2:
                print("1 bench transfer")
                transfer = self.one_transf(df_predictions, True)
            elif free_transf == 1 and worst_XI_nr == 0 and worst_bench_nr <= 2:
                maketransf = False
                print("Save a transfer")
                transfer = pd.DataFrame(columns=['player','points','replacement','rep_points','points_gained'], dtype=object)
            else:
                print("1 normal transfer")
                transfer = self.one_transf(df_predictions, False)

        # Keep only the columns we want and rename them
        self.team_full = self.team_full[['web_name','id','element_type','form','points_per_game',f'predicted_points_{next_GW}','predicted_points_mean',f'predicted_points_{next_GW}_raw','predicted_points_mean_raw']]
        self.team_full.columns = ['web_name','id','element_type','form','points_per_game',f'predicted_points_{next_GW}','predicted_points_mean',f'predicted_points_{next_GW}_raw','predicted_points_mean_raw']

        if maketransf:
            # Take out replaced player and print who to transfer
            for index, row in transfer.iterrows():
                self.team_full = self.team_full[self.team_full.id != int(row['player'])]
                print('Transfer out:',id_name[int(row['player'])],'and transfer in:',id_name[int(row['replacement'])],'for an extra',round(row['points_gained'],2),'points.')
            # Add more stuff to the transfer df, make it ready to merge with team_full
            for col in 'web_name','element_type','form','points_per_game',f'predicted_points_{next_GW}','predicted_points_mean',f'predicted_points_{next_GW}_raw','predicted_points_mean_raw':
                insert(transfer,col,'replacement',df_predictions,col,'id')
            transfer = transfer[['web_name','replacement','element_type','form','points_per_game',f'predicted_points_{next_GW}','predicted_points_mean',f'predicted_points_{next_GW}_raw','predicted_points_mean_raw']]
            transfer.columns=['web_name','id','element_type','form','points_per_game',f'predicted_points_{next_GW}','predicted_points_mean',f'predicted_points_{next_GW}_raw','predicted_points_mean_raw']
        
            frames=[self.team_full,transfer]
            self.team_full = pd.concat(frames)
            self.team_full = self.team_full.sort_values('element_type', ascending = True)
            self.team_full.index = range(len(self.team_full))
            
        self.team_full = self.team_full.sort_values('predicted_points_mean', ascending = False)
        self.team_full=round(self.team_full,2)
        
    
    def choose_best_team(self):
        # The best player is the captain and the second best is the vice captain
        team_full = self.team_full.copy()
        team_full = team_full.sort_values(f'predicted_points_{next_GW}', ascending=False)
        cap, vice = team_full.iloc[0]['web_name'],team_full.iloc[1]['web_name']
        print('The suggested captain is:',cap,'with an expected',team_full.iloc[0][f'predicted_points_{next_GW}'],' points, and vice captain:',vice,
              'with an expected',team_full.iloc[1][f'predicted_points_{next_GW}'],'points. (Or',team_full.iloc[0][f'predicted_points_{next_GW}_raw'],'and',
              team_full.iloc[1][f'predicted_points_{next_GW}_raw'],'unscaled points).')

        # The 3 worst players are on the bench. Take out GKs
        team_full=team_full[team_full.id!=int(self.bench_GK.id)]
        team_outfield=team_full[team_full.element_type!=1]
        team_outfield_XI = team_outfield.head(10)
        team_outfield_bench = team_outfield.tail(3)
        
        formation_dict = dict(team_outfield['element_type'].value_counts())
        formation = str(formation_dict[2])+str(formation_dict[3])+str(formation_dict[4])
        legal_formations = ['541','532','451','442','433','352','343']
        if formation not in legal_formations:
            form_dict = {}
            for form in legal_formations:
                df_def = team_outfield[team_outfield.element_type == 2].head(int(form[0]))
                df_mid = team_outfield[team_outfield.element_type == 3].head(int(form[1]))
                df_atk = team_outfield[team_outfield.element_type == 4].head(int(form[2]))
                form_dict[form]=df_def[f'predicted_points_{next_GW}'].sum()+df_mid[f'predicted_points_{next_GW}'].sum()+df_atk[f'predicted_points_{next_GW}'].sum()
            best_formation = max(form_dict, key=form_dict.get)
            df_def = team_outfield[team_outfield.element_type == 2].head(int(best_formation[0]))
            df_mid = team_outfield[team_outfield.element_type == 3].head(int(best_formation[1]))
            df_atk = team_outfield[team_outfield.element_type == 4].head(int(best_formation[2]))
            team_outfield_XI = pd.concat([df_def,df_mid,df_atk])
            team_outfield_bench = team_outfield.copy()
            for index, player in team_outfield_XI.iterrows():
                team_outfield_bench = team_outfield_bench[team_outfield_bench['id'] != player['id']]
        
        
        team_outfield_bench = team_outfield_bench.sort_values(f'predicted_points_{next_GW}_raw', ascending=False)
        bench_list = list(team_outfield_bench['web_name'])
        print('The bench players should be:',*bench_list, sep=' ')

        # Get the total points for the team
        #team_full = team_outfield_XI.append(team_full[team_full.element_type==1])
        team_full = pd.concat([team_outfield_XI,team_full[team_full.element_type==1]])
        team_full = team_full.sort_values(f'predicted_points_{next_GW}_raw', ascending=False)
        team_full.index = range(len(team_full))
        team_full.at[0,f'predicted_points_{next_GW}_raw']=team_full.at[0,f'predicted_points_{next_GW}_raw']*2
        total_points = round(team_full[f'predicted_points_{next_GW}_raw'].sum(),2)
        print('The total expected points for the team is:',total_points,'.')
        
        return total_points
        
    def update_history(self,team_id,total_points):
        # Add the expected points to the team history and update with last GW's points
        if os.path.isfile(f'ml_team/history_{season}.csv'):
            history_old = pd.read_csv(f'ml_team/history_{season}.csv')
        else:
            d = json.loads(requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/history/').text)
            df = json_normalize(d)
            lol = list(df.current)
            fail = lol[0]
            history_old = json_normalize(fail)
            history_old['predicted_points']=0
            history_old['points - predicted_points']=0

        d = json.loads(requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/history/').text)
        df = json_normalize(d)
        lol = list(df.current)
        fail = lol[0]
        history_new = json_normalize(fail)

        history_new[['predicted_points','points - predicted_points']]=history_old[['predicted_points','points - predicted_points']]
        history_new=history_new[['event','bank','event_transfers','event_transfers_cost','overall_rank','points_on_bench',
                                 'rank','rank_sort','total_points','value','points','predicted_points','points - predicted_points']]
        new_row= pd.DataFrame([[next_GW]+[0]*(len(history_new.columns)-3)+[total_points]+[0]], columns=list(history_new.columns))
        new_row.set_index('event',inplace = True)
        history_new.set_index('event',inplace = True)
        history_new.at[last_GW,'points - predicted_points']=round(history_new.at[last_GW,'points']-history_new.at[last_GW,'predicted_points'],2)

        #history_new = history_new.append(new_row)
        history_new = pd.concat([history_new,new_row])
        history_new.reset_index(inplace=True)

        history_new.to_csv(f'ml_team/history_{season}.csv', index=False)


        
    def one_transf(self, df_predictions, bench_transfer):
        bank = self.bank
        if bench_transfer:
            team = self.bench_outfield.copy()
        else:
            team = self.team_XI.copy()
        change1 = pd.DataFrame(columns=['player','points','replacement','rep_points','points_gained'], dtype=object)
        for index, row in team.iterrows():
            money = row['selling_price'] + bank
            pred_possible = df_predictions[df_predictions.element_type == row['element_type']]
            pred_possible = pred_possible[pred_possible.now_cost <= money]
            # Find the cheapest best new bench player
            if bench_transfer:
                pred_possible = pred_possible[pred_possible.predicted_points_mean > 0.5]
                pred_possible = pred_possible[pred_possible.nail_last5 > 0.5]
                i=39
                bench2 = pred_possible[pred_possible.now_cost < i]
                while len(bench2)==0 and i <= money:
                    i=i+1
                    bench2 = pred_possible[pred_possible.now_cost < i]
                pred_possible = bench2.copy()
            
            ppt_temp = self.players_per_team.copy()
            ppt_temp[row['team']]=ppt_temp[row['team']]-1
            for team_nr, number in ppt_temp.items():
                if number == 3:
                    pred_possible = pred_possible[pred_possible.team != team_nr]
            change1.loc[len(change1)] = [row['id'],row['predicted_points_mean'],pred_possible.iloc[0]['id'],
                                         pred_possible.iloc[0]['predicted_points_mean'],pred_possible.iloc[0]['predicted_points_mean']-row['predicted_points_mean']]
        change1=change1.sort_values('points_gained', ascending=False)
        # Take out GK if injured
        injured=team[team['predicted_points_mean']==0]
        injured=injured.sort_values('element_type', ascending=True)
        if len(injured)>0:
            if int(injured.head(1)['element_type'])==1:
                transfer_id = int(injured.head(1)['id'])
                transfer = change1[change1['player']==transfer_id]
            else:
                transfer = change1.head(1)
        else:
            transfer = change1.head(1)
        return transfer

    def two_transf(self,df_predictions,low_price,high_price):
        team = self.team_XI.copy()
        bank = self.bank
        
        # Add the best replacements for all pairs of players in the first XI
        change2 = pd.DataFrame(columns=['player1','points1','player2','points2','sum_prev_points',
                                        'rep1','rep_points1','rep2','rep_points2','sum_rep_points','points_gained'], dtype=object)

        # Loop over full team
        for index, row1 in team.iterrows():
            money1 = row1['selling_price'] + bank
            pred_possible1 = df_predictions[df_predictions.element_type == row1['element_type']]
            ppt_temp = self.players_per_team.copy()
            ppt_temp[row1['team']]=ppt_temp[row1['team']]-1
            team_removed = team[team['id'] != row1['id']]
            # Loop over full team with one player removed
            for index, row2 in team_removed.iterrows():
                money = row2['selling_price'] + money1 #Total money
                pred_possible2 = df_predictions[df_predictions.element_type == row2['element_type']]
                ppt_temp[row2['team']]=ppt_temp[row2['team']]-1
                for team_nr, number in ppt_temp.items():
                    if number == 3:
                        pred_possible1 = pred_possible1[pred_possible1.team != team_nr]
                        pred_possible2 = pred_possible2[pred_possible2.team != team_nr]
                pred_possible1 = pred_possible1[pred_possible1.now_cost <= money - low_price]
                
                # Make catagorgy for every price
                pred_cat1 = pd.DataFrame()
                for i in np.arange(high_price,low_price,-1):
                    dft=pred_possible1.loc[pred_possible1.now_cost<=i]
                    dft=dft.loc[pred_possible1.now_cost>i-1]
                    #pred_cat1 = pred_cat1.append(dft.iloc[0:1])
                    pred_cat1 = pd.concat([pred_cat1,dft.iloc[0:1]])
                
                for index, rep1 in pred_cat1.iterrows():
                    money2 = money - rep1['now_cost']
                    pred_possible22 = pred_possible2[pred_possible2.now_cost <= money2]
                    pred_possible22 = pred_possible22[pred_possible22['id'] != rep1['id']]
                    ppt_temp[rep1['team']] = ppt_temp[rep1['team']]+1
                    # Remove players from teams we have three players from
                    if ppt_temp[rep1['team']]>2.5:
                        pred_possible22 = pred_possible22[pred_possible22.team != rep1['team']]
                    if len(pred_possible22) > 0.5:
                        change2.loc[len(change2)] = [row1['id'],row1['predicted_points_mean'],row2['id'],row2['predicted_points_mean'],
                                                     row1['predicted_points_mean']+row2['predicted_points_mean'],rep1['id'],rep1['predicted_points_mean'],
                                                    pred_possible22.iloc[0]['id'],pred_possible22.iloc[0]['predicted_points_mean'],
                                                    rep1['predicted_points_mean']+pred_possible22.iloc[0]['predicted_points_mean'],
                                                    rep1['predicted_points_mean']+pred_possible22.iloc[0]['predicted_points_mean']-(row1['predicted_points_mean']+row2['predicted_points_mean'])]
                                                    
        change2=change2.sort_values('points_gained', ascending=False)
        transfer1 = change2[['player1','points1','rep1','rep_points1']].head(1)
        transfer2 = change2[['player2','points2','rep2','rep_points2']].head(1)
        for df in transfer1, transfer2:
            df.columns=['player','points','replacement','rep_points']
        frames = [transfer1,transfer2]
        transfer = pd.concat(frames)
        transfer['points_gained']=transfer.rep_points-transfer.points
        transfer.index=range(len(transfer))
        return transfer

    def one_transf1bench(self, df_predictions):
        team = self.team_XI.copy()
        bench = self.bench_outfield.copy()
        bank = self.bank
        ppt_temp = self.players_per_team.copy()
        
        # Add the best replacements for bench + outfield player combo
        change_form = pd.DataFrame(columns=['player','points','bench_player','bench_points',
                                        'new_bench','new_b_points','new_player','new_points','points_gained'], dtype=object)
        outfield = team[team.element_type !=1]
        for index, row1 in outfield.iterrows(): # Outfield players
            money1 = row1['selling_price'] + bank
            pred_possible1 = df_predictions[df_predictions.element_type == row1['element_type']]

            # Find the cheapest best new bench player
            pred_possible1 = pred_possible1[pred_possible1.predicted_points_mean > 0.5]
            pred_possible1 = pred_possible1[pred_possible1.nail > 0.5]
            i=39
            bench2 = pred_possible1[pred_possible1.now_cost < i]
            while len(bench2)==0:
                i=i+1
                bench2 = pred_possible1[pred_possible1.now_cost < i]
            pred_possible1 = bench2.copy()

            #ppt_temp = players_per_team.copy()
            ppt_temp[row1['team']]=ppt_temp[row1['team']]-1
            team_removed = team[team['id'] != row1['id']]
            for index, row2 in bench.iterrows(): # Bench players
                money = row2['selling_price'] + money1 #Total money
                pred_possible2 = df_predictions[df_predictions.element_type == row2['element_type']]
                ppt_temp[row2['team']]=ppt_temp[row2['team']]-1
                for team_nr, number in ppt_temp.items():
                    if number == 3:
                        pred_possible1 = pred_possible1[pred_possible1.team != team_nr]
                        pred_possible2 = pred_possible2[pred_possible2.team != team_nr]
                # Simply choose the best available bench player and then the best affordable outfield player
                rep1 = pred_possible1.iloc[0]
                money2 = money - rep1['now_cost']
                pred_possible22 = pred_possible2[pred_possible2.now_cost <= money2]
                pred_possible22 = pred_possible22[pred_possible22['id'] != rep1['id']]
                if ppt_temp[rep1['team']]==2:
                    pred_possible22 = pred_possible22[pred_possible22.team != rep1['team']]
                change_form.loc[len(change_form)] = [row1['id'],row1['predicted_points_mean'],row2['id'],row2['predicted_points_mean'],
                                             rep1['id'],rep1['predicted_points_mean'],
                                            pred_possible22.iloc[0]['id'],pred_possible22.iloc[0]['predicted_points_mean'],
                                            pred_possible22.iloc[0]['predicted_points_mean']-row1['predicted_points_mean']]
        change_form=change_form.sort_values('points_gained', ascending=False)
        transfer1 = change_form[['player','points','new_bench','new_b_points']].head(1)
        transfer2 = change_form[['bench_player','bench_points','new_player','new_points']].head(1)
        for df in transfer1, transfer2:
            df.columns=['player','points','replacement','rep_points']
        frames = [transfer1,transfer2]
        transfer = pd.concat(frames)
        transfer['points_gained']=transfer.rep_predicted_points_mean-transfer.predicted_points_mean
        transfer.index=range(len(transfer))
        return transfer    