#PBS -q gshort
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu02/marino
#PBS -o output.txt
#PBS -e error.txt
export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )
source GenUnfold/bin/activate
cd GenerativeUnfolding

src train params/inn.yaml $1
