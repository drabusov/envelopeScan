srun --time=06:00:00 --pty bash

cd /tmp

cp /lustre/nyx/bhs/drabusov/tmp-drabusov.tgz
tar -xzf tmp-drabusov.tgz
mv /tmp/tmp/drabusov/ /tmp/drabusov

rm tmp-drabusov.tgz
rm -r /tmp/tmp

. "/tmp/drabusov/miniconda2/etc/profile.d/conda.sh"
conda activate py3
