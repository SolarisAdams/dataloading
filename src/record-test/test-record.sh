for i in {0..4}
do
    sudo /home/Adama/clear_cache.sh
    /home/Adama/Envs/dataloading/bin/python -m torch.distributed.launch --nproc_per_node=4  /home/Adama/dataloading/src/record-test/full-test-distributed.py --arch ${i} --mode full --num_worker 4
    # /home/Adama/Envs/dataloading/bin/python -m torch.distributed.launch /home/Adama/dataloading/src/pytorch/full-test-distributed.py --arch ${i} --mode train --num_worker 16
done