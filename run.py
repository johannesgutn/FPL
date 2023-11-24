import asyncio
from main_functions import data_main, ml_main, team_main
from credentials import email, password, team_id

# Gather data and perform machine learning 
# Only necessary once per round
#if not os.path.isfile(f'ml_data/ml_data_{season}_{last_GW}.csv'):
if __name__=='__main__':
    data_main()

    ml_main()
    # Figure out the best change for you team
    asyncio.run(team_main(email, password, team_id))