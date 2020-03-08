import torch

class mydataloader(torch.utils.data.DataLoader):
    def __iter__(self):
        if self.num_workers == 0:
            return torch.utils.data.dataloader._SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

class _MultiProcessingDataLoaderIter(torch.utils.data.dataloader._MultiProcessingDataLoaderIter):

    def _next_data(self):
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            out_index = []

            for index in range(self._rcvd_idx, self._send_idx):
                if index in self._task_info and len(self._task_info[index]) == 2:
                    out_index.append(index)
                    if len(out_index) == 2:
                        break
            if len(out_index) == 2:
                data1 = self._task_info.pop(out_index[0])[1]
                data2 = self._task_info.pop(out_index[1])[1]
                tmp_idx = self._rcvd_idx
                while tmp_idx not in self._task_info:
                    tmp_idx += 1
                
                self._rcvd_idx = tmp_idx
                self._try_put_index()
                self._try_put_index()
                # if isinstance(data, ExceptionWrapper):
                #     data.reraise()
                # print(data1, data2)

                return [torch.stack([data1[0],data2[0]],dim=0), torch.stack([data1[1],data2[1]],dim=0)]



            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1

            # if self._dataset_kind == _DatasetKind.Iterable:
            #     # Check for _IterableDatasetStopIteration
            #     if isinstance(data, _utils.worker._IterableDatasetStopIteration):
            #         self._shutdown_worker(data.worker_id)
            #         self._try_put_index()
            #         continue

            
                # store out-of-order samples
            self._task_info[idx] += (data,)






