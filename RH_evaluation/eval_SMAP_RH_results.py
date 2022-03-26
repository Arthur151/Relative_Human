import torch
import numpy as np
import os
import json
from itertools import product

# /path/to/SMAP_RH_results.npz
results_path = '/home/yusun/data_drive/evaluation_results/Relative_results/zip_files/SMAP_RH_results.npz'
# /path/to/Relative_human
RH_dir = '/home/yusun/data_drive/dataset/Relative_human'

def l2_error(j1, j2):
    return np.linalg.norm(j1 - j2, 2)

def _calc_relative_depth_error_weak_(pred_depths, depth_ids, reorganize_idx, age_gts=None, matched_mask=None):
    depth_ids = depth_ids.to(pred_depths.device)
    depth_ids_vmask = depth_ids != -1
    pred_depths_valid = pred_depths[depth_ids_vmask]
    valid_inds = reorganize_idx[depth_ids_vmask]
    depth_ids = depth_ids[depth_ids_vmask]
    age_gts = age_gts[depth_ids_vmask]
    error_dict = {'eq': [], 'cd': [], 'fd':[], 'eq_age': [], 'cd_age': [], 'fd_age':[]}
    error_each_age = {age_type:[] for age_type in relative_age_types}
    for b_ind in torch.unique(valid_inds):
        sample_inds = valid_inds == b_ind
        if matched_mask is not None:
            sample_inds *= matched_mask[depth_ids_vmask]
        did_num = sample_inds.sum()
        if did_num > 1:
            pred_depths_sample = pred_depths_valid[sample_inds]
            triu_mask = torch.triu(torch.ones(did_num, did_num), diagonal=1).bool()
            dist_mat = (pred_depths_sample.unsqueeze(0).repeat(did_num, 1) - pred_depths_sample.unsqueeze(1).repeat(1,did_num))[triu_mask]
            did_mat = (depth_ids[sample_inds].unsqueeze(0).repeat(did_num, 1) - depth_ids[sample_inds].unsqueeze(1).repeat(1,did_num))[triu_mask]
            
            error_dict['eq'].append(dist_mat[did_mat==0])
            error_dict['cd'].append(dist_mat[did_mat<0])
            error_dict['fd'].append(dist_mat[did_mat>0])
            if age_gts is not None:
                age_sample = age_gts[sample_inds]
                age_mat = torch.cat([age_sample.unsqueeze(0).repeat(did_num, 1).unsqueeze(-1), age_sample.unsqueeze(1).repeat(1, did_num).unsqueeze(-1)], -1)[triu_mask]
                error_dict['eq_age'].append(age_mat[did_mat==0])
                error_dict['cd_age'].append(age_mat[did_mat<0])
                error_dict['fd_age'].append(age_mat[did_mat>0])
            # error_dict['all'].append([len(eq_dists), len(cd_dists), len(fd_dists)]) 
            # error_dict['correct'].append([(torch.abs(eq_dists)<thresh).sum().item(), (cd_dists<-thresh).sum().item(), (fd_dists>thresh).sum().item()])

    return error_dict

relative_depth_types = ['eq', 'cd', 'fd']
relative_age_types = ['adult', 'teen', 'kid', 'baby']


BK_19 = {
    'Head_top': 0, 'Nose': 1, 'Neck': 2, 'L_Eye': 3, 'R_Eye': 4, 'L_Shoulder': 5, 'R_Shoulder': 6, 'L_Elbow': 7, 'R_Elbow': 8, 'L_Wrist': 9, 'R_Wrist': 10,\
    'L_Hip': 11, 'R_Hip': 12, 'L_Knee':13, 'R_Knee':14,'L_Ankle':15, 'R_Ankle':16,'L_BigToe':17, 'R_BigToe':18
}

OCHuman_19 = {
    'R_Shoulder':0, 'R_Elbow':1, 'R_Wrist':2, 'L_Shoulder':3, 'L_Elbow':4, 'L_Wrist':5, \
    'R_Hip': 6, 'R_Knee':7, 'R_Ankle':8, 'L_Hip':9, 'L_Knee':10, 'L_Ankle':11, 'Head_top':12, 'Neck':13,\
    'R_Ear':14, 'L_Ear':15, 'Nose':16, 'R_Eye':17, 'L_Eye':18
    }

Crowdpose_14 = {"L_Shoulder":0, "R_Shoulder":1, "L_Elbow":2, "R_Elbow":3, "L_Wrist":4, "R_Wrist":5,\
     "L_Hip":6, "R_Hip":7, "L_Knee":8, "R_Knee":9, "L_Ankle":10, "R_Ankle":11, "Head_top":12, "Neck_LSP":13}

SMAP_15 = {
    'Neck_LSP':0, 'Head_top':1, 'Pelvis':2, 'L_Shoulder':3, 'L_Elbow':4, 'L_Wrist':5, 'L_Hip':6, 'L_Knee':7,\
    'L_Ankle':8, 'R_Shoulder':9, 'R_Elbow':10, 'R_Wrist':11, 'R_Hip':12, 'R_Knee':13, 'R_Ankle':14, 
    }

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format),dtype=np.int)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)

class Evaluate(object):
    def __init__(self):
        super(Evaluate, self).__init__()
        self.set_name = ['val', 'test'][1]
        self.load_gt()
        self.collect_results()
        
        self.kp2d_mapper_BK = joint_mapping(BK_19,Crowdpose_14)
        self.kp2d_mapper_OCH = joint_mapping(OCHuman_19,Crowdpose_14)
        self.kp2d_mapper_MPI = joint_mapping(SMAP_15,Crowdpose_14)
        self.match_kp2ds()
        print('Results on {} set'.format(self.set_name))
        self.calc_error()
    
    def collect_results(self):
        self.results = np.load(results_path,allow_pickle=True)['results'][()]
       
    def collect_results2(self):
        print('loading results..')
        outputs = json.load(open("/home/yusun/data_drive/evaluation_results/Relative_results/zip_files/SMAP_relative_human_results.json", 'r'))['3d_pairs']
        self.results = {}
        for out in outputs:
            img_name = out['image_path']
            self.results[img_name] = {'kp3ds':np.array(out['pred_3d']), 'kp2ds':np.array(out['pred_2d']), 'root_depth':np.array(out['root_d'])}
        img_names = list(self.results.keys())
        for img_name in img_names:
            if img_name not in self.annots:
                del self.results[img_name]
        np.savez('/home/yusun/data_drive/evaluation_results/Relative_results/zip_files/SMAP_RH_results.npz', results=self.results)

    def load_gt(self):
        print('loading gt ..')
        self.annotations = {}
        annot_dir = os.path.join(RH_dir,'{}_annots.npz'.format(self.set_name))
        self.annots = np.load(annot_dir, allow_pickle=True)['annots'][()]

    def no_predictions(self, miss_num):
        self.pr['all'].append(0)
        self.pr['falsePositive'].append(0)
        self.pr['miss'].append(miss_num)

    def match_kp2ds(self):
        self.match_results = {}
        self.pr = {'all':[], 'falsePositive':[], 'miss':[],}
        self.kp2ds = {'gts':{}, 'preds':{}}
        for img_name in self.annots.keys():
            annots = self.annots[img_name]            
            gt_kp2ds = []
            gt_inds = []
            for idx,annot in enumerate(annots):
                vbox = np.array(annot['bbox'])
                fbox = vbox
                if 'kp2d' in annot:
                    if annot['kp2d'] is not None:
                        joint = np.array(annot['kp2d']).reshape((-1,3))
                        invalid_kp_mask = joint[:,2]==0
                        joint[invalid_kp_mask] = -2.
                        joint[:,2] = joint[:,2]>0
                        if len(joint) == 19:
                            is_BK = len(os.path.basename(img_name).replace('.jpg',''))==7
                            if is_BK:
                                joints = joint[self.kp2d_mapper_BK]
                                joints[self.kp2d_mapper_BK==-1] = -2
                            else:
                                joints = joint[self.kp2d_mapper_OCH]
                                joints[self.kp2d_mapper_OCH==-1] = -2
                        elif len(joint) == 14:
                            joints = joint
                        gt_kp2ds.append(joints)
                        gt_inds.append(idx)
            gt_kp2ds = np.array(gt_kp2ds)
            
            if img_name not in self.results:
                print('{} missing from predictions'.format(img_name))
                self.no_predictions(len(gt_kp2ds))
                continue
            results = self.results[img_name]
            pred_kp2ds = results['kp2ds']
            pred_kp2ds[pred_kp2ds[:,:,3]<=0] = -2.
            pred_kp2ds = pred_kp2ds[:,:,:2]
            pred_kp2ds = pred_kp2ds[:,self.kp2d_mapper_MPI]
            pred_kp2ds[:,self.kp2d_mapper_MPI==-1] = -2
            
            valid_gtkps = gt_kp2ds[:,:,2]>0
            valid_person = valid_gtkps.sum(-1)>0
            valid_gtkps = valid_gtkps[valid_person]
            gt_kp2ds = gt_kp2ds[valid_person]

            valid_predkps = pred_kp2ds[:,:,0]>0
            valid_person = valid_predkps.sum(-1)>0
            valid_predkps = valid_predkps[valid_person]
            pred_kp2ds = pred_kp2ds[valid_person]
            if len(pred_kp2ds)==0:
                print('no prediction after processed')
                self.no_predictions(len(gt_kp2ds))
                continue

            assert len(gt_kp2ds)>0, print('no GT')
            bestMatch, falsePositives, misses = match_2d_greedy(pred_kp2ds, gt_kp2ds[:,:,:2], valid_gtkps, valid_predkps, imgPath=img_name)
            
            if len(bestMatch)>0:
                pred_ids, gt_ids = bestMatch[:,0], bestMatch[:,1]
                self.kp2ds['gts'][img_name] = gt_kp2ds[gt_ids,:,:2]
                self.kp2ds['preds'][img_name] = pred_kp2ds[pred_ids]
                bestMatch[:,1] = np.array([gt_inds[ind] for ind in gt_ids])
            self.match_results[img_name] = bestMatch
            self.pr['all'].append(len(pred_kp2ds))
            self.pr['falsePositive'].append(len(falsePositives))
            self.pr['miss'].append(len(misses))

        all_precision, all_recall, all_f1_score = compute_prf1(sum(self.pr['all']), sum(self.pr['miss']), sum(self.pr['falsePositive']))
        
        print('Precision: {} | Recall: {} | F1 score: {}'.format(all_precision, all_recall, all_f1_score))
        

    def calc_error(self):
        self.mPCKh = []
        depth_relative = {'eq': [], 'cd': [], 'fd':[], 'eq_age': [], 'cd_age': [], 'fd_age':[]}
        for img_name, match_mat in self.match_results.items():
            if len(match_mat)==0:
                continue
            pred_ids, gt_ids = match_mat[:,0], match_mat[:,1]
            annots = self.annots[img_name]
            results = self.results[img_name]
            depth_ids = torch.from_numpy(np.array([annots[ind]['depth_id'] for ind in gt_ids]))
            pred_depth = torch.from_numpy(np.array([results['root_depth'][ind] for ind in pred_ids]))
            age_ids = torch.from_numpy(np.array([annots[ind]['age'] for ind in gt_ids]))
            default_organize_idx = torch.zeros(len(depth_ids))

            # mPCKh = _calc_matched_PCKh_(torch.from_numpy(self.kp2ds['gts'][img_name]).float(), torch.from_numpy(self.kp2ds['preds'][img_name]).float(), torch.ones(len(self.kp2ds['gts'][img_name])).bool())
            # self.mPCKh.append(mPCKh)
            # matched_mask = mPCKh > args().matching_pckh_thresh
            matched_mask = torch.ones(len(depth_ids)).bool()
            relative_depth_errors = _calc_relative_depth_error_weak_(pred_depth, depth_ids, default_organize_idx, age_ids, matched_mask=matched_mask.cpu())
            for dr_type in relative_depth_types:
                depth_relative[dr_type] += relative_depth_errors[dr_type]
                depth_relative[dr_type+'_age'] += relative_depth_errors[dr_type+'_age']
        # SMAP mPCKh really bad
        # all_mPCKh = torch.cat(self.mPCKh).mean()
        # print('mPCKh_0.6: {:.2f}'.format(all_mPCKh * 100))
        eval_results = get_results(depth_relative)
        for key, item in eval_results.items():
            print('{}: {:.2f}'.format(key, float(item.item())*100))


def get_results(depth_relative, dr_thresh=0.2):
    eval_results = {}
    eq_dists = torch.cat(depth_relative['eq'], 0)
    cd_dists = torch.cat(depth_relative['cd'], 0)
    fd_dists = torch.cat(depth_relative['fd'], 0)
    eq_age_ids = torch.cat(depth_relative['eq_age'], 0)
    cd_age_ids = torch.cat(depth_relative['cd_age'], 0)
    fd_age_ids = torch.cat(depth_relative['fd_age'], 0)
    dr_age_ids = torch.cat([eq_age_ids, cd_age_ids, fd_age_ids], 0)
    dr_all = np.array([len(eq_dists), len(cd_dists), len(fd_dists)])
    dr_corrects = [torch.abs(eq_dists)<dr_thresh, cd_dists<-dr_thresh, fd_dists>dr_thresh]
    print('Thresh: {} | Equal {:.2f} close {:.2f} far {:.2f}'.format(dr_thresh, dr_corrects[0].sum().item() / dr_all[0] * 100, \
                                        dr_corrects[1].sum().item() / dr_all[1] * 100, dr_corrects[2].sum().item() / dr_all[2] * 100))
    dr_corrects = torch.cat(dr_corrects,0)
    eval_results['PCRD_{}'.format(dr_thresh)] = dr_corrects.sum() / dr_all.sum()
    for age_ind, age_name in enumerate(relative_age_types):
        age_mask = (dr_age_ids == age_ind).sum(-1).bool()
        if age_mask.sum()>0:
            eval_results['PCRD_{}_{}'.format(dr_thresh, age_name)] = dr_corrects[age_mask].sum() / age_mask.sum()
    return eval_results

def get_bbx_overlap(p1, p2, imgpath, baseline=None):
    min_p1 = np.min(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p1 = np.max(p1, axis=0)
    max_p2 = np.max(p2, axis=0)

    bb1 = {}
    bb2 = {}

    bb1['x1'] = min_p1[0]
    bb1['x2'] = max_p1[0]
    bb1['y1'] = min_p1[1]
    bb1['y2'] = max_p1[1]
    bb2['x1'] = min_p2[0]
    bb2['x2'] = max_p2[0]
    bb2['y1'] = min_p2[1]
    bb2['y2'] = max_p2[1]

    try:
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']
    except BaseException:
        print(bb1, bb2)
        logging.fatal('why')

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def scatter2d(gtkp, openpose, imgPath, debug_path, baseline):
    import cv2
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
    img = cv2.imread(imgPath)
    img = img[:, :, ::-1]

    colors = cm.tab20c(np.linspace(0, 1, 25))
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.imshow(img)
    ax2.imshow(img)

    nkps = 24

    for pNum in range(len(openpose)):
        for i in range(nkps):
            ax.scatter(openpose[pNum][i, 0], openpose[pNum]
                       [i, 1], c=colors[i], s=0.05)

    for pNum in range(len(gtkp)):
        for i in range(nkps):
            ax2.scatter(gtkp[pNum][i, 0], gtkp[pNum]
                    [i, 1], c=colors[i], s=0.05)

    if not (imgPath is None):
        savename = imgPath.split('/')[-1]
        savename = savename.replace('.pkl', '.jpg')
        plt.savefig(os.path.join(debug_path, baseline.lower() + savename))
        plt.close('all')

def compute_prf1(count, miss, fp):
    if count == 0:
        return 0, 0, 0
    all_tp = count - miss
    all_fp = fp
    all_fn = miss
    all_f1_score = round(all_tp / (all_tp + 0.5 * (all_fp + all_fn)), 2)
    all_recall = round(all_tp / (all_tp + all_fn), 2)
    all_precision = round(all_tp / (all_tp + all_fp), 2)
    return all_precision, all_recall, all_f1_score


def match_2d_greedy(
        pred_kps,
        gtkp,
        valid_gtkps, valid_predkps, 
        imgPath=None,
        baseline=None,
        iou_thresh=0.05,
        valid=None,
        ind=-1):
    '''
    matches groundtruth keypoints to the detection by considering all possible matchings.
    :return: best possible matching, a list of tuples, where each tuple corresponds to one match of pred_person.to gt_person.
            the order within one tuple is as follows (idx_pred_kps, idx_gt_kps)
    '''
    predList = np.arange(len(pred_kps))
    gtList = np.arange(len(gtkp))
    # get all pairs of elements in pred_kps, gtkp
    # all combinations of 2 elements from l1 and l2
    combs = list(product(predList, gtList))

    errors_per_pair = {}
    errors_per_pair_list = []
    for comb in combs:
        vmask = valid_gtkps[comb[1]] * valid_predkps[comb[0]]
        if vmask.sum()>0:
            assert vmask.sum()>0, print('no valid points')
            errors_per_pair[str(comb)] = l2_error(
                pred_kps[comb[0]][vmask, :2], gtkp[comb[1]][vmask, :2])
            
        else:
            print('no valid points')
            errors_per_pair[str(comb)] = 1000.
        errors_per_pair_list.append(errors_per_pair[str(comb)])

    gtAssigned = np.zeros((len(gtkp),), dtype=bool)
    opAssigned = np.zeros((len(pred_kps),), dtype=bool)
    errors_per_pair_list = np.array(errors_per_pair_list)

    bestMatch = []
    excludedGtBecauseInvalid = []
    falsePositiveCounter = 0
    while np.sum(gtAssigned) < len(gtAssigned) and np.sum(
            opAssigned) + falsePositiveCounter < len(pred_kps):
        found = False
        falsePositive = False
        while not(found):
            if sum(np.inf == errors_per_pair_list) == len(
                    errors_per_pair_list):
                print('something went wrong here')

            minIdx = np.argmin(errors_per_pair_list)
            minComb = combs[minIdx]
            # compute IOU
            iou = get_bbx_overlap(
                pred_kps[minComb[0]], gtkp[minComb[1]], imgPath, baseline)
            # if neither prediction nor ground truth has been matched before and iou
            # is larger than threshold
            if not(opAssigned[minComb[0]]) and not(
                    gtAssigned[minComb[1]]) and iou >= iou_thresh:
                #print(imgPath + ': found matching')
                found = True
                errors_per_pair_list[minIdx] = np.inf
            else:
                errors_per_pair_list[minIdx] = np.inf
                # if errors_per_pair_list[minIdx] >
                # matching_threshold*headBboxs[combs[minIdx][1]]:
                if iou < iou_thresh:
                    #print(
                    #   imgPath + ': false positive detected using threshold')
                    found = True
                    falsePositive = True
                    falsePositiveCounter += 1

        # if ground truth of combination is valid keep the match, else exclude
        # gt from matching
        if not(valid is None):
            if valid[minComb[1]]:
                if not falsePositive:
                    bestMatch.append(minComb)
                    opAssigned[minComb[0]] = True
                    gtAssigned[minComb[1]] = True
            else:
                gtAssigned[minComb[1]] = True
                excludedGtBecauseInvalid.append(minComb[1])

        elif not falsePositive:
            # same as above but without checking for valid
            bestMatch.append(minComb)
            opAssigned[minComb[0]] = True
            gtAssigned[minComb[1]] = True

    bestMatch = np.array(bestMatch)
    # add false positives and false negatives to the matching
    # find which elements have been successfully assigned
    opAssigned = []
    gtAssigned = []
    for pair in bestMatch:
        opAssigned.append(pair[0])
        gtAssigned.append(pair[1])
    opAssigned.sort()
    gtAssigned.sort()

    falsePositives = []
    misses = []

    # handle false positives
    opIds = np.arange(len(pred_kps))
    # returns values of oIds that are not in opAssigned
    notAssignedIds = np.setdiff1d(opIds, opAssigned)
    for notAssignedId in notAssignedIds:
        falsePositives.append(notAssignedId)
    gtIds = np.arange(len(gtList))
    # returns values of gtIds that are not in gtAssigned
    notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned)

    # handle false negatives/misses
    for notAssignedIdGt in notAssignedIdsGt:
        if not(valid is None):  # if using the new matching
            if valid[notAssignedIdGt]:
                #print(imgPath + ': miss')
                misses.append(notAssignedIdGt)
            else:
                excludedGtBecauseInvalid.append(notAssignedIdGt)
        else:
            #print(imgPath + ': miss')
            misses.append(notAssignedIdGt)

    return bestMatch, falsePositives, misses  # tuples are (idx_pred_kps, idx_gt_kps)

if __name__ == '__main__':
    Evaluate()