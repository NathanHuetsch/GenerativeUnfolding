#PBS -q gshort
#PBS -l nodes=1:ppn=1:gpus=1:gshort
#PBS -l walltime=40:00:00
#PBS -d /remote/gpu07/huetsch
#PBS -o output.txt
#PBS -e error.txt
export CUDA_VISIBLE_DEVICES=$(cat $PBS_GPUFILE | sed s/.*-gpu// )
module load anaconda/3.0
module load cuda/11.8
source activate /remote/gpu07/huetsch/madgraph
cd GenerativeUnfolding

#pip install --editable .
memennto train $1
