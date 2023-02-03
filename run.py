from main_functions import *

# Check whether the newest data has already been downloaded.  If not it downloads it
if not os.path.isfile(f'player_data/player_data_{season}_{last_GW}.csv'):
    data_main()


email = "example@gmail.com"
password = "password"
team_id = 123456789 # To get the team ID go to "Pick Team" then "Gameweek History", and then the ID is in the URL

# Suggests the optimal changes for your team
asyncio.run(team_main(email, password,team_id))



