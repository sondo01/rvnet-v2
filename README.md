# rv_net


## Steps
1. Download all the files from this repository
2. navigate to the rv_net folder in your terminal. Then, replicate the anaconda environment by typing the following in a command prompt
<code> conda env create -f tensorflow_env.yml </code>
3. Run the <code>2_3_1_HARPS_Linear_FC_CNN_June10_2023.ipynb</code>  script


## Modifications for making .yml files to work on Other computers
1. Create the .yml file without builds
   <code>conda env export --no-builds > my_environment.yml</code>
2. In addition, this may not work on Windows devices. There does not seem to be an easy way to automatically remove OS-specific packages, but there is a way to see which packages were specifically installed using <code>conda install</code> so we could use that to remove all other packages from the .yml file.
   <code>conda env export --from-history</code>


## Other important things
Allocation: <code> sched_mit_andrewv </code>

directory: <code> /nfs/mkilab001/Groups/Andrew_Vanderburg </code>

loading anaconda <code> module load anaconda3/2021.11 </code>

extracting a ton of tar.gz files into one directory (directory_name) <code> cat *.tar | tar -xvf - -i -C directory_name </code>


