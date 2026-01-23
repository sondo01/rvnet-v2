import os
from dace_query.sun import Sun
output_directory='/Volumes/LACIE SETUP/RV_NET'
# Sun.download_public_release_all('2015','12', output_directory=output_directory,output_filename='release_all_2015-12.tar.gz')

Sun.download_public_release_timeseries(output_directory=output_directory, output_filename='public_release_timeseries.tar.gz')
# # 1. Configuration
# start_date = "2015-07-29"
# end_date = "2018-05-24"
# output_dir = f"data_{start_date}_to_{end_date}