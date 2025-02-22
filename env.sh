spack load python@3.12.5/t72fixe
spack load cuda@12.4.1/f3kmmeb
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source venv/bin/activate

export NCU=$(spack location -i cuda@12.6.3)/bin/ncu
export H1="srun -n 1 --gres=gpu:H100:1"
export H="srun -n 1 --gres=gpu:H100:1"
