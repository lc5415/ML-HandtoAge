#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=48gb:ngpus=2:gpu_type=RTX6000
#PBS -N CHALEOptimalLR
module load anaconda3/personal

cd /rdsgpfs/general/user/lc5415/home/ML-HtoA

python3 Scripts/OptimalLR.py -wd 0 -fn '0CHALEOptLR.csv'
python3 Scripts/OptimalLR.py -wd 0.1 -fn '01CHALEOptLR.csv'
python3 Scripts/OptimalLR.py -wd 0.01 -fn '001CHALEOptLR.csv'
python3 Scripts/OptimalLR.py -wd 0.001 -fn '0001CHALEOptLR.csv'
python3 Scripts/OptimalLR.py -wd 0.0001 -fn '00001CHALEOptLR.csv'
