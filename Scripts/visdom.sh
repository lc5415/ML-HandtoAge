#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=1:mem=4gb

module load anaconda3/personal

cd /rdsgpfs/general/user/lc5415/home/MLProject/ML-HandtoAge

python -m visdom.server -port 8686

python3 visdomTest.py
