#!/bin/bash

MPIRUNOPTS='-bysocket -bind-to-socket -display-map -report-bindings'

# RUN VERSION
VERSION=1

# Necessary to know QEVERSION
if [ -n "$1" ]; then
  . $1
else
  . run_default.list
fi

# doublecheck this!
if [ -n "$2" ]; then
  FINALDIR=/gpfs/scratch/userinternal/acv0/Filippo_scratch/VALIDATION-${QEVERSION}-`date "+%Y%m%d"`.$2
else
  FINALDIR=/gpfs/scratch/userinternal/acv0/Filippo_scratch/VALIDATION-${QEVERSION}-`date "+%Y%m%d"`
fi

# Necessary to fill properly FINALDIR
if [ -n "$1" ]; then
  . $1
else
  . run_default.list
fi

mkdir -p ${FINALDIR}
mkdir -p ${FINALDIR}/OUTPUT

for (( j=1; j <= $number_of_exes; j++ ))
do
  for (( i=1; i <= $number_of_tests; i++ ))
  do
    # Create running directory...
    RUNNINGDIR=${FINALDIR}/RUN-${j}-${i}
    mkdir -p ${RUNNINGDIR}
    cp -R ${inputfiles[$i]}/* ${RUNNINGDIR}/
    
    FILE=`pwd`/submit.${j}.${i}.pbs

    # Create parameter array...
    index=0
    for tokens in ${parameters[$i]}
    do
       decoded_params[index]=${tokens}
       ((index++))
    done
 
    SCRIPT=$(cat <<EOF
#!/bin/bash
#PBS -l select=${decoded_params[0]}:mpiprocs=${decoded_params[1]}:ncpus=${decoded_params[2]}:ngpus=2:mem=40gb
#PBS -l walltime=${decoded_params[3]}
#PBS -q parallel
#PBS -A cinstaff
#PBS -N QE5-VALIDATION
#PBS -j oe

. /cineca/prod/environment/module/3.1.6/none/init/bash
module load profile/advanced
module load intel/co-2011.6.233--binary openmpi/1.4.4--intel--co-2011.6.233--binary
module load cuda/4.2.9
module load python
sleep 5
cd ${RUNNINGDIR}

export OMP_NUM_THREADS=${decoded_params[4]}
export MKL_NUM_THREADS=${decoded_params[4]}
export PHI_DGEMM_SPLIT=${decoded_params[5]}
export PHI_ZGEMM_SPLIT=${decoded_params[6]}
sleep 1
# Clean previous run
rm -rf save CRASH
sleep 1
#Copy executable
cp ${EXEDIR}/${exes[$j]}.x ./
sleep 1
# Run
mpirun -v -np ${decoded_params[7]} -npernode ${decoded_params[8]} ${MPIRUNOPTS} ./${exes[$j]}.x -input ${decoded_params[10]}.in ${exe_params[$i]} | tee out.validation.${exes[$j]}.${decoded_params[9]}.MPI-${decoded_params[7]}.OMP-${decoded_params[4]}.SPLIT-${decoded_params[5]}.${decoded_params[10]}.${VERSION}
sleep 1
# Copy output in the OUTPUT directory
cp out.validation.* ${FINALDIR}/OUTPUT/
#mkdir ${FINALDIR}/OUTPUT/profiling.RUN-${j}-${i}
#mv *.csv  ${FINALDIR}/OUTPUT/profiling.RUN-${j}-${i}/
# Exit successfully
exit 0
EOF
)
    echo "${SCRIPT}" > ${FILE}
    qsub ${FILE}
  done
done