#PBS -l walltime=20:00:00
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=2:gpu_type=RTX6000
#PBS -N AdamOptimalCHALE
module load anaconda3/personal

cd /rdsgpfs/general/user/lc5415/home/ML-HtoA

python3 Scripts/train.py -wd 0.01 -rf 'AdamOptiLRWD' --step-size 10
