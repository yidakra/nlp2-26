#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -p staging
#SBATCH -o jupyter-notebook-job.out

source ~/.bashrc
conda activate mt

PORT=`shuf -i 5000-5999 -n 1`
LOGIN_HOST=${SLURM_SUBMIT_HOST}-pub.snellius.surf.nl
BATCH_HOST=$(hostname)

echo "To connect to the notebook type the following command from your local terminal:"
echo "ssh -N -J ${USER}@${LOGIN_HOST} ${USER}@${BATCH_HOST} -L ${PORT}:localhost:${PORT}"
echo
echo "After connection is established in your local browser go to the address:"
echo "http://localhost:${PORT}"

jupyter notebook --no-browser --port $PORT

