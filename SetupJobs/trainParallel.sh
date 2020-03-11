#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=2:gpu_type=RTX6000
#PBS -N trainToy
#PBS -J 1-4
module load anaconda3/personal


cd /rdsgpfs/general/user/lc5415/home/MLProject/ML-HandtoAge

arch = @PBS_ARRAY_INDEX
python3 train.py 1
