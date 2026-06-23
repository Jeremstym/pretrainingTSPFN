#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --nodelist=punk
#SBATCH --job-name=CauKer-generating
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-23:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err

uname -a
nvidia-smi
ulimit -n 4096

# poetry run python /home/stympopper/CauKer/CauKer.py -N 100000 -L 512 -F 4 -P 6 -M 18 -O /data/stympopper/CauKer/CauKer100K5124CH.arrow
# poetry run python /home/stympopper/CauKer/CauKer.py -N 100000 -L 512 -F 3 -P 6 -M 18 -O /data/stympopper/CauKer/CauKer100K5123CH.arrow
# poetry run python /home/stympopper/CauKer/CauKer.py -N 100000 -L 512 -F 2 -P 6 -M 18 -O /data/stympopper/CauKer/CauKer100K5122CH.arrow
# poetry run python /home/stympopper/CauKer/CauKer.py -N 100000 -L 512 -F 1 -P 6 -M 18 -O /data/stympopper/CauKer/CauKer100K5121CH.arrow
poetry run python /home/stympopper/CauKer/CauKer.py -N 100000 -L 512 -F 5 -P 7 -M 20 -O /data/stympopper/CauKer/CauKer100K5125CH.arrow
poetry run python /home/stympopper/CauKer/CauKer.py -N 100000 -L 512 -F 6 -P 7 -M 20 -O /data/stympopper/CauKer/CauKer100K5126CH.arrow
poetry run python /home/stympopper/CauKer/CauKer.py -N 100000 -L 512 -F 7 -P 7 -M 20 -O /data/stympopper/CauKer/CauKer100K5127CH.arrow
