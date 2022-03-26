import torch
import numpy as np
import os

# /path/to/SMAP_RH_results.npz
results_path = '/home/yusun/data_drive/evaluation_results/Relative_results/zip_files/3DMPPE_RH_results.npz'
# /path/to/Relative_human
RH_dir = '/home/yusun/data_drive/dataset/Relative_human'

relative_depth_types = ['eq', 'cd', 'fd']
relative_age_types = ['adult', 'teen', 'kid', 'baby']

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

class Evaluate(object):
    def __init__(self):
        super(Evaluate, self).__init__()
        self.set_name = ['val', 'test'][1]
        self.load_gt()
        self.collect_results()
        
        print('Results on {} set'.format(self.set_name))
        self.calc_error()
    
    def collect_results(self):
        self.root_depths = np.load(results_path,allow_pickle=True)['results'][()]
    
    def collect_results2(self):
        print('loading results..')
        self.root_depths = np.load("/home/yusun/data_drive/evaluation_results/Relative_results/zip_files/3DMPPE_Relative_human_{}_root_depths.npz".format(self.set_name),allow_pickle=True)['results'][()]
        img_names = list(self.root_depths.keys())
        for img_name in img_names:
            if img_name not in self.annots:
                del self.root_depths[img_name]
        np.savez(results_path, results=self.root_depths)

    def load_gt(self):
        print('loading gt ..')
        self.annotations = {}
        annot_dir = os.path.join(RH_dir,'{}_annots.npz'.format(self.set_name))
        self.annots = np.load(annot_dir, allow_pickle=True)['annots'][()]
        
    def calc_error(self):
        depth_relative = {'eq': [], 'cd': [], 'fd':[], 'eq_age': [], 'cd_age': [], 'fd_age':[]}
        for img_name, annots in self.annots.items():
            gt_ids, pred_depth = self.root_depths[img_name]
            pred_depth = torch.from_numpy(np.array(pred_depth))
            depth_ids = torch.from_numpy(np.array([annots[ind]['depth_id'] for ind in gt_ids]))
            age_ids = torch.from_numpy(np.array([annots[ind]['age'] for ind in gt_ids]))
            default_organize_idx = torch.zeros(len(depth_ids))
            matched_mask = torch.ones(len(depth_ids)).bool()
            relative_depth_errors = _calc_relative_depth_error_weak_(pred_depth, depth_ids, default_organize_idx, age_ids, matched_mask=matched_mask)
            for dr_type in relative_depth_types:
                depth_relative[dr_type] += relative_depth_errors[dr_type]
                depth_relative[dr_type+'_age'] += relative_depth_errors[dr_type+'_age']

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


if __name__ == '__main__':
    Evaluate()