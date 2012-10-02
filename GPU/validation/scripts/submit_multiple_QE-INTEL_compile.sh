#!/bin/bash

# Module cuda required to fill some env variables
#. /cineca/prod/environment/module/3.1.6/none/init/bash
# module load cuda/4.0

# General variables
TMPDIR='/tmp/'

PWDDIR=`pwd`

# Necessary to know QEVERSION
if [ -n "$1" ]; then
  . $1
else
  . compile_default.list
fi

# doublecheck this!
if [ -n "$2" ]; then
  FINALDIR=/gpfs/scratch/userinternal/acv0/Filippo_scratch/exes-${QEVERSION}-`date "+%Y%m%d"`.$2
else
  FINALDIR=/gpfs/scratch/userinternal/acv0/Filippo_scratch/exes-${QEVERSION}-`date "+%Y%m%d"`
fi

# Necessary to fill properly FINALDIR
if [ -n "$1" ]; then
  . $1
else
  . compile_PRACE.list
fi

mkdir -p ${FINALDIR}

# Compress before copy
#tar zcf espresso-build.tar.gz --exclude="*.svn" ${QEDIR}

for (( i=1; i <= $makescript_lenght; i++ ))
do
    COMPILEDIR=${TMPDIR}tmp-espresso.`date "+%Y%m%d%H%M%S"`.${i}
    FILE=`pwd`/submit.${i}.pbs
    SCRIPT=$(cat <<EOF
#!/bin/bash
#PBS -l walltime=0:20:00
#PBS -l select=1:mpiprocs=1:ncpus=1
#PBS -q debug
#PBS -A cinstaff
#PBS -N QE5-COMPILE
#PBS -j oe
. /cineca/prod/environment/module/3.1.6/none/init/bash
module load profile/advanced
module load intel/co-2011.6.233--binary openmpi/1.4.4--intel--co-2011.6.233--binary
module load cuda/4.2.9
module load python
sleep 5
mkdir -p ${COMPILEDIR}
### #cp -R ${QEDIR}/* ${COMPILEDIR}/
cd ${COMPILEDIR}
cp /gpfs/scratch/userinternal/acv0/Filippo_scratch/espresso-build.tar.gz .
tar zxf espresso-build.tar.gz
cd espresso 
EOF
)
  echo "${SCRIPT}" > ${FILE}
  echo "${makescript[$i]}" >> ${FILE}
  qsub ${FILE}
done
