from main_functions import *
from credentials import *

# Gather data and perform machine learning 
# Only necessary once per round
if not os.path.isfile(f'ml_data/ml_data_{season}_{last_GW}.csv'):
    data_main()

    ml_main()

# Figure out the best change for you team
asyncio.run(team_main(email, password, team_id))