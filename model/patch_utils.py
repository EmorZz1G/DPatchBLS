from einops import rearrange
import torch.nn.functional as F


# According to the length of the sequence and the patch_size, the patch_num is obtained (less than 1)
def get_patch_num(seq_len, patch_size):
    if seq_len % patch_size != 0:
        patch_num = seq_len // patch_size + 1
    else:
        patch_num = seq_len // patch_size
    return patch_num


# Restore the shape of the patches to align with the original series shape and trim the data to its length before the FIX padding was implemented, eliminating the initial padded sections.
def patch2series(x, before_len):
    return rearrange(x, 't p -> (t p)')[-before_len:]


# Adjust the input time series x by padding the beginning if seq_len is not divisible by patch_size, ensuring it can be evenly divided by patch_size.
def patch_fix(x, patch_size, verbose=False):
    seq_len = len(x)
    if seq_len % patch_size != 0:
        tmp = patch_size - (seq_len % patch_size)
        x = F.pad(x, (tmp, 0))
        if verbose:
            print('pad', x.shape)
    x = rearrange(x, '(t p) -> t p', p=patch_size)
    if verbose:
        print('Input', x.shape)
    return x
