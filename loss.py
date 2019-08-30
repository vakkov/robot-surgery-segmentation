import torch
from torch import nn
from torch import einsum
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
import utils
from utils import cuda
import numpy as np

from modules.lovasz import lovasz_hinge, lovasz_softmax, lovasz_softmax_flat
from typing import Iterable, List, Set, Tuple, TypeVar
from scipy.ndimage import distance_transform_edt as distance
from torch.nn.modules.distance import PairwiseDistance


class LossBinary:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss
    # def whoami(self):
    #    print type(self).__name__

class LossStableBCE:
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * binary_xloss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss


class LossMulti:
    def __init__(self, jaccard_weight=0, class_weights=None, num_classes=1):
        if class_weights is not None:
            nll_weight = utils.cuda(
                torch.from_numpy(class_weights.astype(np.float32)))
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        #outputs = F.log_softmax(outputs, dim=1)
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= torch.log((intersection + eps) / (union - intersection + eps)) * self.jaccard_weight
        return loss



class BCELoss(_Loss):
    def __init__(self, per_image=True, from_logits=True):
        super().__init__()
        self.per_image = per_image
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor):
        batch_size = y_pred.size(0)
        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
        else:
            loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')

        if self.per_image:
            return loss.view(batch_size, -1).mean(dim=1)
        return loss.mean()


class FocalLoss(_Loss):
    def __init__(self, gamma, per_image=True, from_logits=True):
        super().__init__()
        self.gamma = float(gamma)
        self.per_image = per_image
        self.from_logits = from_logits

    def forward(self, y_pred: Tensor, y_true: Tensor):
        if self.from_logits:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')

            # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
            invprobs = F.logsigmoid(-y_pred * (y_true * 2 - 1))
            loss = (invprobs * self.gamma).exp() * loss
        else:
            loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')

            # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
            invprobs = -y_pred * (y_true * 2 - 1)
            loss = (invprobs * self.gamma).exp() * loss

        if self.per_image:
            batch_size = y_pred.size(0)
            return loss.view(batch_size, -1).mean(dim=1)

        return loss.mean()


class JaccardLoss(_Loss):
    def __init__(self, per_image=True, from_logits=True, smooth=10):
        super().__init__()
        self.from_logits = from_logits
        self.per_image = per_image
        self.smooth = float(smooth)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        batch_size = y_pred.size(0)

        if self.from_logits:
            y_pred = torch.sigmoid(y_pred)

        if self.per_image:
            y_pred = y_pred.view(batch_size, -1)
            y_true = y_true.view(batch_size, -1)

            intersection = torch.sum(y_pred * y_true, dim=1)
            union = torch.sum(y_pred, dim=1) + torch.sum(y_true, dim=1) - intersection
        else:
            intersection = torch.sum(y_pred * y_true, dim=None)
            union = torch.sum(y_pred, dim=None) + torch.sum(y_true, dim=None) - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


class BCEAndJaccardLoss(_Loss):
    def __init__(self, bce_weight=1, jaccard_weight=1, per_image=True, from_logits=True):
        super().__init__()
        self.bce = BCELoss(per_image=per_image, from_logits=from_logits)
        self.bce_weight = float(bce_weight)
        self.jaccard = JaccardLoss(per_image=per_image, from_logits=from_logits)
        self.jaccard_weight = float(jaccard_weight)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        bce_loss = self.bce(y_pred, y_true)
        iou_loss = self.jaccard(y_pred, y_true)
        return (bce_loss * self.bce_weight + iou_loss * self.jaccard_weight) / (self.bce_weight + self.jaccard_weight)


class FocalAndJaccardLoss(_Loss):
    def __init__(self, focal_weight=1, jaccard_weight=1, per_image=True, from_logits=True):
        super().__init__()
        self.focal = FocalLoss(per_image=per_image, from_logits=from_logits, gamma=2)
        self.focal_weight = float(focal_weight)
        self.jaccard = JaccardLoss(per_image=per_image, from_logits=from_logits)
        self.jaccard_weight = float(jaccard_weight)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        foc_loss = self.focal(y_pred, y_true)
        iou_loss = self.jaccard(y_pred, y_true)
        return (foc_loss * self.focal_weight + iou_loss * self.jaccard_weight) / (self.focal_weight + self.jaccard_weight)


class LovaszHingeLoss(_Loss):
    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, output: Tensor, target: Tensor):
        return lovasz_hinge(output, target, self.per_image, self.ignore)


class BCEAndLovaszLoss(_Loss):
    def __init__(self, bce_weight=1, lovasz_weight=1, per_image=True, from_logits=True):
        super().__init__()
        if not from_logits:
            raise ValueError("This loss operates only on logits")

        self.bce = BCELoss(per_image=per_image, from_logits=from_logits)
        self.bce_weight = float(bce_weight)
        self.lovasz = LovaszHingeLoss(per_image=per_image)
        self.lovasz_weight = float(lovasz_weight)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        bce_loss = self.bce(y_pred, y_true)
        lov_loss = self.lovasz(y_pred, y_true)
        return (bce_loss * self.bce_weight + lov_loss * self.lovasz_weight) / (self.bce_weight + self.lovasz_weight)

# class LovaszSoftmax(_Loss):
#     def __init__(self, classes='present', per_image=True, ignore=None):
#         super().__init__()
#         self.classes = classes
#         self.per_image = per_image
#         self.ignore = ignore

#     def forward(self, output: Tensor, target: Tensor):
#         return lovasz_softmax(output, target, self.per_image, self.ignore)

class LovaszSoftmax(nn.Module):
    def __init__(self, per_image=False):
        super(LovaszSoftmax, self).__init__()
        self.lovasz_softmax = lovasz_softmax
        self.per_image = per_image

    def forward(self, pred, label):
        """
        :param pred:  b, c, h, w
        :param label:  b, h, w
        :return:
        """
        pred = F.softmax(pred, dim=1)
        res = self.lovasz_softmax(pred, label, per_image=self.per_image)
        #print("lovasz_softmax: ", res)
        return res

# class LovaszSoftmax(nn.Module):
#     """
#     Multi-class Lovasz-Softmax loss
#       logits: [B, C, H, W] class logits at each prediction (between -\infty and \infty)
#       labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
#       ignore_index: void class labels
#       only_present: average only on classes present in ground truth
#     """
#     def __init__(self, ignore_index=None, only_present=True):
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.only_present = only_present

#     def forward(self, logits, labels):
#         probas = F.softmax(logits, dim=1)
#         total_loss = 0
#         batch_size = logits.shape[0]
#         for prb, lbl in zip(probas, labels):
#             total_loss += lovasz_softmax_flat(prb, lbl, self.ignore_index, self.only_present)
#         return total_loss / batch_size        


#lyakaap/pytorch-template
class SoftIoULoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    #def to_one_hot(tensor):
    def to_one_hot(tensor, n_classes):
        tensor = tensor.to(device="cuda")
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).to(device='cuda') 
        one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w), 1)
        return one_hot

    def forward(self, logit, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(logit)

        logit = F.softmax(logit, dim=1)
        #pred = pred.cuda()
        #target_onehot = self.to_one_hot(target)
        target_onehot = torch.zeros_like(logit)
        # print("logit ", logit.shape)
        # print("target ", target.shape)
        # print("pred ", pred.shape)
        # print("scores 3 ", scores.shape[3])

        target_onehot.scatter_(1, target.view(logit.shape[0],1,logit.shape[2],logit.shape[3]), 1)  

        # Numerator Product
        inter = logit * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = logit + target_onehot - (logit * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        #return -loss.mean()
        return 1 -loss.mean()



######## SURFACE LOSS ########
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    #C: int = len(seg)
    C = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res



# def one_hot2dist(seg: Tensor) -> Tensor:    
#     C = len(seg)
#     print("seg ", seg)

#     res = torch.zeros_like(seg)
#     for c in range(C):
#         posmask = seg[c].type(torch.ByteTensor)
#         #posmask = seg[c]

#         if posmask.any():
#             negmask = ~posmask
#             #res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
#             res[c] = PairwiseDistance(2).forward(negmask, negmask) * negmask - (PairwiseDistance(2).forward(posmask, posmask) - 1) * posmask
#             #res[c] =  torch.norm(negmask - negmask, 2) * negmask - (torch.norm(posmask - posmask, 2) - 1) * posmask
#     return res
    #return torch.from_numpy(res).cuda()


        # def to_one_hot(tensor, n_classes):
        # #tensor = tensor.to(device="cuda")
        # n, h, w = tensor.size()
        # one_hot = torch.zeros(n, n_classes, h, w).to(device='cuda')
        # one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w), 1)


# class CrossEntropy():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")

#     def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
#         assert simplex(probs) and simplex(target)

#         log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
#         mask: Tensor = target[:, self.idc, ...].type(torch.float32)

#         loss = - einsum("bcwh,bcwh->", mask, log_p)
#         loss /= mask.sum() + 1e-10

#         return loss

# class DiceLoss():
#     def __init__(self, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = kwargs["idc"]
#         print(f"Initialized {self.__class__.__name__} with {kwargs}")

#     def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
#         assert simplex(probs) and simplex(target)

#         pc = probs[:, self.idc, ...].type(torch.float32)
#         tc = target[:, self.idc, ...].type(torch.float32)

#         intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
#         union: Tensor = (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

#         divided: Tensor = 1 - (2 * intersection + 1e-10) / (union + 1e-10)

#         loss = divided.mean()

#         return loss

# class GeneralizedDice(nn.Module):
#     def __init__(self, **kwargs):
#         super(GeneralizedDice, self).__init__()
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         #print("kwargs", kwargs["idc"])
#         self.idc = kwargs["idc"]
#         #print(f"Initialized {self.__class__.__name__} with {kwargs}")

#     # def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
#     #def forward(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
#     def forward(self, probs, target):
def GeneralizedDice(probs, target, idc):
        assert simplex(probs) and simplex(target)

        pc = probs[:, idc, ...].type(torch.float32)
        tc = target[:, idc, ...].type(torch.float32)

        w = torch.Tensor()
        intersection = torch.Tensor()
        union = torch.Tensor()
        divided = torch.Tensor()

        #torch.save(tc, 'targets2.pt')
        #torch.save(pc, "probs2.pt")
        #tc_sum = einsum("bcwh->bc", tc)

        w = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection = w * einsum("bcwh,bcwh->bc", pc, tc)
        union = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)
        #print("divided ", divided)
        loss = divided.mean()

        return loss

def GeneralizedDice2(probs, target, dist_maps, idc, idc2):
        assert simplex(probs) and simplex(target)
        assert not one_hot(dist_maps)


        pc = probs[:, idc, ...].type(torch.float32)
        tc = target[:, idc, ...].type(torch.float32)
        #pc2 = probs[:, idc2, ...].type(torch.float32)
        dc = dist_maps[:, idc, ...].type(torch.float32) 

        w = torch.Tensor()
        intersection = torch.Tensor()
        union = torch.Tensor()
        divided = torch.Tensor()

        #torch.save(tc, 'targets2.pt')
        #torch.save(pc, "probs2.pt")
        #tc_sum = einsum("bcwh->bc", tc)

        w = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection = w * einsum("bcwh,bcwh->bc", pc, tc)
        union = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)
        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)
        #print("divided ", divided)
        loss = divided.mean()
        loss += multipled.mean()

        return loss

def SurfaceLoss(probs, dist_maps, idc):
        assert simplex(probs)
        assert not one_hot(dist_maps)

        # probs[:, 0, ...] = 0
        # dist_maps[:,0, ...] = 0
        # pc = torch.zeros(probs.shape)
        # dc = torch.zeros(dist_maps.shape)

        # pc = pc[0, probs[:, idc, ...]].type(torch.float32)
        # dc = dc[0, dist_maps[:, idc, ...]].type(torch.float32)
        
        pc = probs[:, idc, ...].type(torch.float32)
        dc = dist_maps[:, idc, ...].type(torch.float32)        

        print("pc ", pc.shape)
        print("dc ", dc.shape)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()
        #print("loss: ", loss)

        return loss        

class Combined(nn.Module):
    def __init__(self, **kwargs):
        super(Combined, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs["idc"]
    def forward(self, probs: Tensor, target: Tensor, onehot_labels: Tensor, dist_maps: Tensor) -> Tensor:
        outputs_softmaxes = F.softmax(probs, dim=1)
        
        #with torch.no_grad():
        onehot_labels = cuda(onehot_labels)
        dist_maps = cuda(dist_maps)
        
        #print("onehot_labels ", onehot_labels)
        #region_loss = GeneralizedDice(probs=outputs_softmaxes, target=onehot_labels, idc=[0, 1, 2, 3, 4, 5, 6, 7])
        #surface_loss = SurfaceLoss(probs=outputs_softmaxes, dist_maps=dist_maps, idc=[1, 2, 3, 4, 5, 6, 7])
        region_loss = GeneralizedDice2(probs=outputs_softmaxes, target=onehot_labels, dist_maps=dist_maps, idc=[0, 1, 2, 3, 4, 5, 6, 7], idc2=[1, 2, 3, 4, 5, 6, 7])

        print("region: ", region_loss)
        #print("surface: ", surface_loss)
        #alpha = 0.80
        #total_loss = (alpha*region_loss) + ((1-alpha) * surface_loss)
        #print("total_loss ", total_loss)
        #return total_loss
        return region_loss

        #return GeneralizedDice(probs=outputs_softmaxes, target=onehot_labels, idc=[0, 1]) + SurfaceLoss(probs=outputs_softmaxes, dist_maps=dist_maps, idc=[1]) 

class Combined_Lovasz(nn.Module):
    def __init__(self, **kwargs):
        super(Combined_Lovasz, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = kwargs["idc"]
    def forward(self, probs: Tensor, target: Tensor, onehot_labels: Tensor, dist_maps: Tensor) -> Tensor:
        outputs_softmaxes = F.softmax(probs, dim=1)

        #with torch.no_grad():
        #    onehot_labels = cuda(onehot_labels)
        dist_maps = cuda(dist_maps)

        #print("onehot_labels ", onehot_labels)
        lovasz = lovasz_softmax(outputs_softmaxes, target, per_image=False)
        surface_loss = SurfaceLoss(probs=outputs_softmaxes, dist_maps=dist_maps, idc=[1, 2, 3, 4, 5, 6, 7])

        print("lovasz: ", lovasz)
        print("surface: ", surface_loss)
        #alpha = torch.tensor(0.80, dtype=torch.float32).cuda()
        #remainer = torch.tensor(1.0 - alpha, dtype=torch.float32).cuda()
        #total_loss = (alpha*lovasz).add( (remainer * surface_loss))
        #total_loss = (torch.mm(alpha,lovasz)).add(torch.mm((1.0-alpha), surface_loss))
        alpha = 0.99
        total_loss = (alpha*lovasz) + ((1-alpha) * surface_loss)
        #print("total_loss ", total_loss)
        return total_loss

# class SurfaceLoss():
#     def __init__(self, num_classes, **kwargs):
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         #self.idc: List[int] = kwargs["idc"]
#         self.idc = kwargs.get('idc')
#        #self.num_classes = kwargs.get('num_classes')
#         self.num_classes = num_classes
#         #print(f"Initialized {self.__class__.__name__} with ", **kwargs)

#     #def __call__(self, probs: Tensor, labels: Tensor, _: Tensor) -> Tensor:
#     def __call__(self, probs: Tensor, labels: Tensor) -> Tensor:
#         #print("labels shape ", labels.shape)
#         print("classes: ", self.num_classes)
#         dist_maps = one_hot2dist(class2one_hot(labels, self.num_classes))
#         #dist_maps = class2one_hot(one_hot2dist(labels.detach()))

#         assert simplex(probs)
#         #assert not one_hot(dist_maps)

#         pc = probs[:, self.idc, ...].type(torch.float32)
#         dc = dist_maps[:, self.idc, ...].type(torch.float32)
#         #print("probs ", probs.shape)
#         #print("dist_maps ", dist_maps.shape)

#         #check loss2 DICELoss for these two
#         # pc = probs[:, [0, 1], ...].type(torch.float32)
#         # dc = dist_maps[:, [0, 1], ...].type(torch.float32)

#         multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

#         loss = multipled.mean()

#         return loss