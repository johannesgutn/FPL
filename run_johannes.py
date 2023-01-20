from main_functions import *

# Check if we need to
if not os.path.isfile(f'raw_data/raw_data_{season}_{last_GW}.csv'):
    data_main()

email = 'kafforno@gmail.com'
password = 'silver11.F' 
team_id = 3429906


asyncio.run(team_main(email, password,team_id))