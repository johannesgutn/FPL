from main_functions import *

if not os.path.isfile(f'raw_data/raw_data_{season}_{last_GW}.csv'):
    data_main()


email = "example@gmail.com"
password = "password"
team_id = 123456789

asyncio.run(team_main(email, password,team_id))



