import numpy as np
import torch
from path import Path
from collections import OrderedDict
import cv2


class Compose(object):
    """Composes image transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, depth, intrinsics, height = 240, width =320):
        for t in self.transforms:
            images, depth, intrinsics = t(images, depth, intrinsics, height, width)
        return images, depth, intrinsics


class Normalize(object):
    """Normalizes images"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, depth, intrinsics, height = 240, width =320):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, depth, intrinsics


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor"""

    def __call__(self, images, depth, intrinsics, height = 240, width =320):
        tensors = []
        for im in images:
            im = np.transpose(im, (2, 0, 1))
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, depth, intrinsics



class Scale(object):
    """Scales images and modifies corresponding intrinsics"""

    def __call__(self, images, depth, intrinsics, height = 240, width =320):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)


        out_h = height
        out_w = width

        in_h, in_w, _ = images[0].shape
        x_scaling_int = out_w/in_w
        y_scaling_int = out_h/in_h

        x_scaling = [out_w/im.shape[1] for im in images]
        y_scaling = [out_h/im.shape[0] for im in images]



        output_intrinsics[0][0] *= x_scaling_int
        output_intrinsics[1][1] *= y_scaling_int

        output_intrinsics[0][2] = x_scaling_int*(output_intrinsics[0][2]+0.5)-0.5
        output_intrinsics[1][2] = y_scaling_int*(output_intrinsics[1][2]+0.5)-0.5


        dx_scaling = [out_w/dim.shape[1] for dim in depth]
        dy_scaling = [out_h/dim.shape[0] for dim in depth]


        scaled_images = [cv2.resize(images[i], None, fx=x_scaling[i], fy=y_scaling[i], interpolation = cv2.INTER_LINEAR)  for i in range(len(images))]
        scaled_depth = [cv2.resize(depth[i], None, fx=dx_scaling[i], fy=dy_scaling[i], interpolation = cv2.INTER_LINEAR)  for i in range(len(depth))]
        

        return scaled_images, scaled_depth, output_intrinsics



def compute_errors(gt, pred, valid, print_res=False):
    """Evaluates depth metrics"""

    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)


    res_list = []
    for current_gt, current_pred, current_valid in zip(gt, pred, valid):
        valid_gt = current_gt[current_valid]
        valid_pred = current_pred[current_valid]


        if len(valid_gt) == 0:
            continue
        else:
            thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
            a1 += (thresh < 1.25).float().mean()
            a2 += (thresh < 1.25 ** 2).float().mean()
            a3 += (thresh < 1.25 ** 3).float().mean()

            abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
            if print_res:
                res_list.append(torch.mean(torch.abs(valid_gt - valid_pred)).item())
            abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

            sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    if print_res:
        print(res_list)

    return [metric / batch_size for metric in [abs_rel, abs_diff, sq_rel, a1, a2, a3]]



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)



def reorder_desc(desc,batch_sz): 
    """Reorders Descriptors"""

    b,c,h,w = desc.shape
    desc = desc.view(-1,batch_sz,c,h,w)
    desc = desc.transpose(1,0)
    return desc   


def pose_square(pose):
    """Converts pose matrix of size 3x4 to a square matrix of size 4x4"""

    pose_sh = pose.shape
    if pose_sh[2] ==3:
        pose_row = torch.tensor([0.,0.,0.,1.]) 
        if pose.is_cuda:
            pose_row = pose_row.to(pose.device)
        pose_row = pose_row.repeat(pose_sh[0],pose_sh[1],1,1)
        pose = torch.cat((pose,pose_row),2)

    return pose


def make_symmetric(anc, ref):
    """Makes anchor and reference tensors symmetric"""
    
    if (anc is None) or (ref is None):
        return None
    ancs = anc.shape
    views = torch.stack(ref,0)
    if len(ancs)==3:
        views   = views.view(-1,ancs[1],ancs[2])
    else:
        views   = views.view(-1,anc.shape[1],ancs[2],ancs[3])
    anc_ref = torch.cat((anc, views),0)
    return anc_ref



