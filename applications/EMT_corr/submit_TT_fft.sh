#!/bin/bash

Rsq_list="1.2.3.4.5.6.8.9.10.11.12.13.14.16.17.18.19.20.21.22.24.25.26.27.29.35.40.60.80.100.120.140.180.220.260.320.360.400.520.640.700.800.1000.1400.1800.2200.3072"
conflist="1050 1146 1194 1242 1290 1338 1386 1434 1482 1530 1578 1626 1674 1722 1770 1818 1866 1914 1962 2010"
#Rsq_list="1.3"
#conflist="1050"

for i in $conflist
do
    echo conf:$i
echo "#!/bin/bash	
#SBATCH --job-name=TT
#SBATCH --output=/u/shu1/test/Log/TT_conf$i.out
#SBATCH --partition=gpuA40x4
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=\"scratch\"
#SBATCH --gpus-per-task=1
#SBATCH --account=bbuu-delta-gpu
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 45:00:00

module load fftw
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sw/spack/delta-2022-03/apps/fftw/3.3.10-gcc-11.2.0-ipxfmko/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/sw/spack/delta-2022-03/apps/fftw/3.3.10-gcc-11.2.0-ipxfmko/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/sw/spack/delta-2022-03/apps/fftw/3.3.10-gcc-11.2.0-ipxfmko/include

source /u/shu1/gpt_denn/lib/cgpt/build/source.sh
source ~/.bashrc

srun -N 1 -n 1 python3.9 -u /u/shu1/test/get_TT_2pt.py --mpi 1.1.1.1 --mpi_split 1.1.1.1 --PathConf \"/scratch/bbuu/shu1/conf_nersc/l6464f21b7130m00119m0322a_nersc.$i\" --PathTwoPtOutFolder \"/u/shu1/test/TwoPtOut/\" --confnum $i --Rsq_list $Rsq_list" > ./sbatches/sbatch_TT_conf$i.sh
sbatch ./sbatches/sbatch_TT_conf$i.sh
done
