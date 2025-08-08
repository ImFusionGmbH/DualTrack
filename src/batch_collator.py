import torch 


class BatchCollator:
    def __init__(self, length_key="tracking", pad_keys=[]):
        self.length_key = length_key
        self.pad_keys = pad_keys

    def __call__(self, batch):
        keys = batch[0].keys()
        lengths = [len(batch_i[self.length_key]) for batch_i in batch]
        max_length = max(lengths)

        out = {}

        for key in keys:
            if key not in self.pad_keys:
                out[key] = [batch_i[key] for batch_i in batch]
            else:
                value = []
                for i, length in enumerate(lengths):
                    value_i = batch[i][key]
                    size = [max_length - length]
                    size += value_i.shape[1:]
                    padding = torch.zeros(size, dtype=value_i.dtype)
                    value_i = torch.cat([value_i, padding], dim=0)
                    value.append(value_i)
                out[key] = torch.stack(value, dim=0)

        padding_size = [(max_length - length) for length in lengths]
        out["padding_size"] = padding_size
        return out
