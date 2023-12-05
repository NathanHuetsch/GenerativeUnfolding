#PBS -q a30
#PBS -l nodes=1:ppn=1:gpus=1:a30
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu07/huetsch
#PBS -o output.txt
#PBS -e error.txt
export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )
source venv/bin/activate
cd GenerativeUnfolding

#src train $1
src train params/inn.yaml
