import numpy as np
import cv2
import os
import copy

dataset_dir = '/home/yusun/data_drive/dataset/Relative_human'
image_folder = os.path.join(dataset_dir, 'images')
visualiztion = True

#-----------------------------------------------------------------------------------------#
#                                   data loading utils                                    #
#-----------------------------------------------------------------------------------------#
name_dict_chinese = {
    'depth_id': {0: '最前排', 1: '第二排', 2: '第三排', 3: '第四排', 4: '第五排', 5: '第六排', 6: '第七排', 7: '第八排', 8: '第九排', 9: '第十排', -1: '深度不明'},
    'age': {0: '成年', 1: '青少年', 2: '小孩', 3:'婴幼儿', -1: '年龄不明'},
    'body_type': {0: '正常', 1: '微胖', 2: '胖', 3: '强壮'},
    'occluded_by_others': {0: '无遮挡', 1: '遮挡'},
    'gender': {0: '男', 1: '女', -1:'性别不明'}
}

name_dict = {
    'age': {0: 'adult', 1: 'teenager', 2: 'kid', 3:'baby', -1: 'unknown'},
    'gender': {0: 'male', 1: 'female', -1:'unknown'}
}

def load_annots(dataset_dir, split_name='val'):
    path = os.path.join(dataset_dir, f'{split_name}_annots.npz')
    annots = np.load(path, allow_pickle=True)['annots'][()]
    return annots

def load_image(image_name):
    image_path = os.path.join(image_folder, image_name)
    return cv2.imread(image_path)

def print_annots_info(packed_annots):
    kp2ds, bboxes, meta_info = packed_annots
    print(name_dict)
    print('kp2ds', kp2ds)
    print('bbox:', bboxes)
    print('Depth Layers number:', meta_info[:,0])
    print('Age type:', meta_info[:,1])
    print('Gender:', meta_info[:,2])

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

def determine_skeleton_type(img_name, joint):
    if len(joint) == 19:
        is_BK = len(os.path.basename(img_name).replace('.jpg',''))==7
        if is_BK:
            skeleton = BK_19
        else:
            skeleton = OCHuman_19
    elif len(joint) == 14:
        skeleton = Crowdpose_14
    return skeleton

def prepare_annots(annots, img_name):
    kp2ds, bboxes, meta_info = [], [], []
        
    for ind, annot in enumerate(annots):
        vbox = np.array(annot['bbox']) # rough bbox for visible body parts only
        fbox = np.array(annot['bbox_wb']) if 'bbox_wb' in annot else None # bbox for the whole body
        bboxes.append([vbox, fbox])

        joint, skeleton = None, None
        if 'kp2d' in annot:
            if annot['kp2d'] is not None:
                joint = np.array(annot['kp2d']).reshape((-1,3))
                invalid_kp_mask = joint[:,2]==0
                joint[invalid_kp_mask] = -2.
                joint[:,2] = joint[:,2]>0
                skeleton = determine_skeleton_type(img_name, joint)                     
        kp2ds.append([joint, skeleton])

        meta_info.append([annot['depth_id'], annot['age'], annot['gender']])

    meta_info = np.array(meta_info)
    return (kp2ds, bboxes, meta_info)

#-----------------------------------------------------------------------------------------#
#                                  visualization utils                                    #
#-----------------------------------------------------------------------------------------#

relative_age_types = ['adult', 'teen', 'kid', 'baby']

SMPL_24 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23
    }

SMPL_EXTRA_30 = {
    'Nose':24, 'R_Eye':25, 'L_Eye':26, 'R_Ear': 27, 'L_Ear':28, \
    'L_BigToe':29, 'L_SmallToe': 30, 'L_Heel':31, 'R_BigToe':32,'R_SmallToe':33, 'R_Heel':34, \
    'L_Hand_thumb':35, 'L_Hand_index': 36, 'L_Hand_middle':37, 'L_Hand_ring':38, 'L_Hand_pinky':39, \
    'R_Hand_thumb':40, 'R_Hand_index':41,'R_Hand_middle':42, 'R_Hand_ring':43, 'R_Hand_pinky': 44, \
    'R_Hip': 45, 'L_Hip':46, 'Neck_LSP':47, 'Head_top':48, 'Pelvis':49, 'Thorax_MPII':50, \
    'Spine_H36M':51, 'Jaw_H36M':52, 'Head':53
    }

SMPL_ALL_54 = {**SMPL_24, **SMPL_EXTRA_30}

smpl24_connMat = np.array([0,1, 0,2, 0,3, 1,4,4,7,7,10, 2,5,5,8,8,11, 3,6,6,9,9,12,12,15, 12,13,13,16,16,18,18,20,20,22, 12,14,14,17,17,19,19,21,21,23]).reshape(-1, 2)
# joint connection relationship for two hands, two feet, face, tow lsp hips, neck and head
All54_connMat = np.concatenate([smpl24_connMat, np.array([
    [20, 35], [20, 36], [20, 37], [20, 38], [20, 39], [21, 40], [21, 41], [21, 42], [21, 43], [21, 44], \
    [7, 29], [7, 31], [29, 30], [8, 32], [8, 34], [32, 33], \
    [24, 25], [25, 27], [24, 26], [26, 28], \
    [45, 49], [45, 5], [46, 49], [46, 4], \
    [47, 16], [47, 17], [47, 48], [47, 50], [51, 49], [51, 50], [12, 50], [52, 47], [52, 12], [53, 47], [53, 12] 
    ]) ], 0)

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format),dtype=np.int)*-1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)


def draw_skeleton(image, pts, bones=None, cm=None, label_kp_order=False, r=6):
    for i,pt in enumerate(pts):
        if len(pt)>1:
            if pt[0]>0 and pt[1]>0:
                image = cv2.circle(image,(int(pt[0]), int(pt[1])),r,cm,-1)
                if label_kp_order and i in bones:
                    img=cv2.putText(image,str(i),(int(pt[0]), int(pt[1])),cv2.FONT_HERSHEY_COMPLEX,1,(255,215,0),1)
    
    if bones is not None:
        set_colors = np.array([cm for i in range(len(bones))]).astype(np.int)
    
        bones = np.concatenate([bones,set_colors],1).tolist()
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            if (pa>0).all() and (pb>0).all():
                xa,ya,xb,yb = int(pa[0]),int(pa[1]),int(pb[0]),int(pb[1])
                image = cv2.line(image,(xa,ya),(xb,yb),(int(line[2]), int(line[3]), int(line[4])), r)
    return image

color_table = np.array([
    [0.4, 0.6, 1], # blue
    [0.8, 0.7, 1], # pink
    [0.1, 0.9, 1], # cyan
    [0.8, 0.9, 1], # gray
    [1, 0.6, 0.4], # orange
    [1, 0.7, 0.8], # rose
    [1, 0.9, 0.1], # Yellow
    [1, 0.9, 0.8], # skin
    [0.9, 1, 1],   # light blue
    [0.9, 0.7, 0.4], # brown
    [0.8, 0.7, 1], # purple
    [0.8, 0.9, 1], # light blue 2
    [0.9, 0.3, 0.1], # red
    [0.7, 1, 0.6],   # green
    [0.7, 0.4, 0.6], # dark purple
    [0.3, 0.5, 1], # deep blue
])[:,::-1] * 255

def draw_skeleton_multiperson(image, pts_group, skeletons, colors=None):
    if colors is None:
        colors = np.array([color_table[i%len(color_table)] for i in range(len(pts_group))])
    for pts, skeleton, color in zip(pts_group, skeletons, colors):
        if pts is None:
            continue
        kp2d_mapping = joint_mapping(skeleton, SMPL_ALL_54)
        kp2d = pts[kp2d_mapping]
        kp2d[kp2d_mapping == -1] = -2
        image = draw_skeleton(image, kp2d, bones=All54_connMat, cm=color)
    return image

def visualize_2d(image, packed_annots):
    kp2ds_info, bboxes, meta_info = packed_annots
    kp2ds, skeletons = [kp[0] for kp in kp2ds_info], [kp[1] for kp in kp2ds_info] 
    vboxes = np.array([bbox[0] for bbox in bboxes]).astype(np.int)

    colors = np.array([color_table[i%len(color_table)] for i in range(len(kp2ds))])
    skeleton_image = draw_skeleton_multiperson(image, kp2ds, skeletons, colors=colors)
    for ind, vbox in enumerate(vboxes):
        #cv2.rectangle(image, tuple(vbox[:2]), tuple(vbox[2:]), tuple(colors[ind]), 2)
        depth, age, gender = meta_info[ind]
        info = 'D{}, {}, {}'.format(depth, name_dict['age'][age], name_dict['gender'][gender])
        cv2.putText(image, info, tuple(vbox[:2]+np.array([0,24])), cv2.FONT_HERSHEY_COMPLEX, 1, tuple(colors[ind]), 2)
    cv2.imshow('skeleton', skeleton_image)
    cv2.waitKey(0)

def calc_crop_bbox(kp2ds, expand_ratio=np.array([1.1,1.2])):
    vis_masks = (kp2ds>0).sum(-1)>1
    bboxes = np.zeros((len(kp2ds), 3, 2), dtype=np.int)
    for ind, kp2d in enumerate(kp2ds):
        left = kp2d[vis_masks[ind],0].min()
        right = kp2d[vis_masks[ind],0].max()
        top = kp2d[vis_masks[ind],1].min()
        bottom = kp2d[vis_masks[ind],1].max()
        center = np.array([(left+right)/2, (top+bottom)/2])
        width_height = np.array([right-left,bottom-top])*expand_ratio
        ltrbxy = np.array([center-width_height/2, center+width_height/2, center])
        bboxes[ind] = ltrbxy
    return bboxes

def visualize_3d(image, packed_annots, interactive_show=True):
    try:
        import vedo
    except:
        os.system('pip install vedo')
        import vedo
    plt = vedo.Plotter(bg=[240,255,255], axes=1, offscreen=not interactive_show)
    kp2ds_info, bboxes, meta_info = packed_annots
    if kp2ds_info[0][0] is None:
        return 
    kp2ds = np.array([kp[0] for kp in kp2ds_info])
    skeletons = [kp[1] for kp in kp2ds_info] 
    bboxes = calc_crop_bbox(kp2ds)
    colors = np.array([color_table[i%len(color_table)] for i in range(len(kp2ds))])

    height = image.shape[0]
    depth_interval = 300
    pic = vedo.Picture(image[:,:,::-1])
    pic.z(-(meta_info[:,0].max()+1)*depth_interval)
    plt += pic

    for ind, bbox in enumerate(bboxes):
        (l, t), (r, b), (cx, cy) = bbox[0], bbox[1], bbox[2]
        crop_image_patch = copy.deepcopy(image[t:b,l:r])
        kp2d = kp2ds[ind]
        kp2d[:,0] -= l
        kp2d[:,1] -= t
        skeleton_image = draw_skeleton_multiperson(crop_image_patch, [kp2d], [skeletons[ind]], colors=[colors[ind]])
        depth, age, gender = meta_info[ind]
        #info = '{}, {}'.format(name_dict['age'][age], name_dict['gender'][gender])
        #cv2.putText(skeleton_image, info, (30,30), cv2.FONT_HERSHEY_COMPLEX, 1, tuple(colors[ind]), 2)
        pic = vedo.Picture(skeleton_image[:,:,::-1])
        pic.z(-depth*depth_interval).x(cx).y(height-cy)
        plt += pic
    plt.show()
    plt.close()

def main():
    annots = load_annots(dataset_dir, split_name='train')
    for example_image_name in ['6182308.jpg']: #list(annots.keys())
        print(example_image_name)
        packed_annots = prepare_annots(annots[example_image_name], example_image_name)
        #print_annots_info(packed_annots)

        if visualiztion:
            image = load_image(example_image_name)
            visualize_2d(copy.deepcopy(image), packed_annots)
            visualize_3d(copy.deepcopy(image), packed_annots)

if __name__ == '__main__':
    main()