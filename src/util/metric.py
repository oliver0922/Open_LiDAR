
'''IoU'''
import torch
import numpy as np

UNKNOWN_ID = 255
NO_FEATURE_ID = 256


def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        return confusion[:num_classes, :num_classes]

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom


def evaluate(pred_ids, gt_ids, stdout=True, dataset='nuscenes'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')

    if 'nuscenes' in dataset:
        CLASS_LABELS = ('barrier', 'bicycle', 'bus', 'car', 'construction vehicle', 'motorcycle', 'person', 'traffic cone',
                      'trailer', 'truck', 'drivable surface', 'other flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')
        pred_ids = pred_ids[gt_ids!=255]
        gt_ids = gt_ids[gt_ids!=255]
    elif 'waymo' in dataset:
        CLASS_LABELS = ('car', 'truck', 'bus', 'other vehicle', 'motorcyclist', 'bicyclist', 'pedestrian', 'sign', 'traffic light', 
                        'pole', 'construction cone', 'bicycle', 'motorcycle', 'building', 'vegetation', 'tree trunk', 'curb', 'road',
                        'lane marker', 'other ground', 'walkable', 'sidewalk')
        pred_ids = pred_ids[gt_ids!=255]
        gt_ids = gt_ids[gt_ids!=255]
        gt_ids -= 1
    else:
        raise NotImplementedError

    N_CLASSES = len(CLASS_LABELS)
    
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    mean_iou /= N_CLASSES
    mean_acc /= N_CLASSES
    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                if 'matterport' in dataset:
                    print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))

                else:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2]))
            except:
                print(label_name + ' error!')
                continue
        print('Mean IoU', mean_iou)
        print('Mean Acc', mean_acc)
    return mean_iou



class AverageMeter():
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def soft_intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3, 4])
    
    output = output.float()

    output[target == ignore_index,:] = ignore_index
    soft_intersection = torch.where(torch.sum(output == target.view(-1,1), dim=1))[0]
    top1_output = output[:,0].float()
    top1_output[soft_intersection] = target[soft_intersection].float()
    intersection = output[top1_output == target]
    
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(top1_output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()