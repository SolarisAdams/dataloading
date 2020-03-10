for i in {0..4}
do
    sudo /home/Adama/clear_cache.sh
    /home/Adama/Envs/dataloading/bin/python -m torch.distributed.launch --nproc_per_node=4  /home/Adama/dataloading/src/pytorch/full-test-distributed.py --arch ${i} --mode full --num_worker 4
    # /home/Adama/Envs/dataloading/bin/python -m torch.distributed.launch /home/Adama/dataloading/src/pytorch/full-test-distributed.py --arch ${i} --mode train --num_worker 16
done
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py train 0 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py load 0 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py full 0 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py train 1 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py load 1 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py full 1 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py train 2 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py load 2 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py full 2 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py train 3 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py load 3 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py full 3 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py train 4 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py load 4 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py full 4 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py train 0 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py load 0 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py full 0 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py train 1 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py load 1 16
# sleep 10
# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/pytorch/full-test.py full 1 16
# sleep 10

# /home/Adama/Envs/dataloading/bin/python /home/Adama/dataloading/src/postprocess.py