#!/bin/bash

# Change number of tasks, amount of memory and time limit according to your needs

#SBATCH -n 1
#SBATCH --time=3:0:0
#SBATCH --mem=4G
#SBATCH -J jupyter
#SBATCH --output=%x-%j.out

# Load Jupyter Notebook
module load notebook

## Get host IP
host_ip=$(hostname -i)

# Select free random port
# jupyter_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()');
jupyter_port=$((10000 + RANDOM % 50000))

# Print tunnel command
cat 1>&2 <<END
Open a new terminal window in Mac and Linux or a local terminal in Mobaxterm

Copy and paste the command bellow in new terminal to create a tunnel to the cluster

ssh -J $USER@bava.cs.huji.ac.il -L $jupyter_port:$host_ip:$jupyter_port $USER@moriah-gw.cs.huji.ac.il

After creating the tunnel copy and paste the URL that Jupyter Notebook created in a local browser.

END

# Uncomment and enter path of code
cd /sci/nosnap/arieljaffe/sagi.nathan/adni_code

# Start Jupyter
jupyter-notebook --no-browser --ip=0.0.0.0 --port=$jupyter_port