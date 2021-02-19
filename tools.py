import torch


def clip_by_tensor(t, batchsize):

    t = t.float()
    t_min = torch.zeros(batchsize, 2)
    t_min += 0.001
    t_max = torch.ones(batchsize, 2)
    t_max -= 0.001

    t_min = t_min.cuda()
    t_max = t_max.cuda()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result