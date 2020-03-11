#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=1:mem=4gb

module load anaconda3/personal

cd /rdsgpfs/general/user/lc5415/home/ML-HtoA

python -m visdom.server -port 8686


