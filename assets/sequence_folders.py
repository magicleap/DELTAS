import numpy as np
import cv2
from path import Path
import random
import os 
import pickle
import torch


def load_as_float(path):
    """Loads image"""
    im =  cv2.imread(path)
    im =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) 
    return im


class SequenceFolder(torch.utils.data.Dataset):
    """Creates a pickle file for ScanNet scene loading, and corresponding dataloader"""

    def __init__(self, root, seed=None, ttype='test.txt', sequence_length=2, sequence_gap = 20, transform=None, height = 240, width = 320):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)

        scene_list_path = ttype
        self.scene_list_path = scene_list_path[:-4]
        fold_root = 'scans_test' 
        scenes = [self.root/fold_root/folder[:-1] for folder in open(scene_list_path)]
        self.ttype = ttype
        self.scenes = sorted(scenes)

        self.width = width
        self.height = height

        self.transform = transform
        file_pickle = self.scene_list_path+ '_len_'+str(sequence_length)+ '_gap_'+str(sequence_gap)+'.pickle'
        if os.path.exists(file_pickle):
            with open(file_pickle, 'rb') as handle:
                sequence_set = pickle.load(handle)
                self.samples = sequence_set
        else:
            self.crawl_folders(sequence_length,sequence_gap)



    def crawl_folders(self, sequence_length,sequence_gap):
        sequence_set = []
        isc = 0
        for scene in self.scenes:
            #print(isc, len(self.scenes))
            isc+=1
            frames = os.listdir(os.path.join(scene, "color"))
            frames = [int(os.path.splitext(frame)[0]) for frame in frames]
            frames =  sorted(frames)
            poses = [np.loadtxt(os.path.join(scene, "pose", "%d.txt" % frame)) for frame in frames]


            with open(os.path.join(scene, "%s.txt" % scene.split('/')[-1])) as info_f:
                info = [line.rstrip().split(' = ') for line in info_f]
                info = {key:value for key, value in info}

                intrinsics = np.asarray([
                    [float(info['fx_color']), 0, float(info['mx_color'])],
                    [0, float(info['fy_color']), float(info['my_color'])],
                    [0, 0, 1]
                ]).astype(np.float32)


            if len(frames) < sequence_gap*sequence_length:
                continue


            path_split = scene.split('/')

            for i in range(len(frames)):
 
                img = os.path.join(scene, "color", "%d.jpg" % i)
                depth = os.path.join(scene, "depth", "%d.png" % i)

                pose_tgt = poses[i]

                do_nan_tgt = False
                nan_pose_tgt = np.sum(np.isnan(pose_tgt) | np.isinf(pose_tgt))
                if nan_pose_tgt>0:
                    do_nan_tgt = True


                save_name = "%d_" % i 

                sample = {'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': depth, 'ref_depths': [], 'ref_imgs': [], 'ref_poses': [], 'path': []}                
                sample['path'] = os.path.join(scene , img.name[:-4])

                if i < sequence_gap:
                    shifts = list(range(i,i+(sequence_length-1)*sequence_gap+1,sequence_gap))
                    shifts.remove(i) #.pop(i)
                elif i >= len(frames)- sequence_gap:
                    shifts = list(range(i,len(frames),sequence_gap))
                    shifts = list(range(i-(sequence_length-1)*sequence_gap,i+1,sequence_gap))
                    shifts.remove(i)
                else:
                    if sequence_length%2 == 1:
                        demi_length = sequence_length//2
                        if (i>=demi_length*sequence_gap) and (i<len(frames)- demi_length*sequence_gap):
                            shifts = list(range(i- (demi_length)*sequence_gap, i+(demi_length)*sequence_gap+1,sequence_gap))
                        elif i<demi_length*sequence_gap:
                            
                            diff_demi = (demi_length-i//sequence_gap)
                            shifts = list(range(i- (demi_length-diff_demi)*sequence_gap, i+(demi_length+diff_demi)*sequence_gap+1,sequence_gap))
                        elif i>=len(frames)- demi_length*sequence_gap:
                           
                            diff_demi = (demi_length-(len(frames)-i-1)//sequence_gap)
                            shifts = list(range(i- (demi_length+diff_demi)*sequence_gap, i+(demi_length-diff_demi)*sequence_gap+1,sequence_gap))
                        else:
                            print('Error')
                        shifts.remove(i)
                    else:
                        #2 scenarios
                        demi_length = sequence_length//2
                        if (i >= demi_length*sequence_gap) and (i<len(frames)- demi_length*sequence_gap):
                            shifts = list(range(i- demi_length*sequence_gap, i+(demi_length-1)*sequence_gap+1,sequence_gap))
                        elif i<demi_length*sequence_gap:    
                            diff_demi = (demi_length-i//sequence_gap)
                            shifts = list(range(i- (demi_length-diff_demi)*sequence_gap, i+(demi_length+diff_demi-1)*sequence_gap+1,sequence_gap))
                        elif i>=len(frames)- demi_length*sequence_gap:
                           
                            diff_demi = (demi_length-(len(frames)-i-1)//sequence_gap)
                            shifts = list(range(i- (demi_length+diff_demi-1)*sequence_gap, i+(demi_length-diff_demi)*sequence_gap+1,sequence_gap))
                        else:
                            print('Error')
                        shifts.remove(i)

     
                
                do_nan = False
                for j in shifts:

                    pose_src = poses[j]
                    pose_rel =  np.linalg.inv(pose_src) @ pose_tgt
                    pose = pose_rel[:3,:].reshape((1,3,4)).astype(np.float32)
                    sample['ref_poses'].append(pose)


                    sample['ref_imgs'].append(os.path.join(scene, "color", "%d.jpg" % j))
                    sample['ref_depths'].append(os.path.join(scene, "depth", "%d.png" % j))

                    nan_pose = np.sum(np.isnan(pose)) + np.sum(np.isinf(pose))
                    if nan_pose>0:
                        do_nan = True
           

                if not do_nan_tgt and not do_nan:
                   sequence_set.append(sample)

        
        file_pickle = self.scene_list_path+ '_len_'+str(sequence_length)+ '_gap_'+str(sequence_gap)+'.pickle'
        with open(file_pickle, 'wb') as handle:
            pickle.dump(sequence_set, handle, protocol=pickle.HIGHEST_PROTOCOL)


        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        tgt_depth = cv2.imread(sample['tgt_depth'],-1).astype(np.float32)/1000

        ref_poses = sample['ref_poses']

        ref_depths = [cv2.imread(depth_img,-1).astype(np.float32)/1000 for depth_img in sample['ref_depths']]
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]


        if self.transform is not None:
            imgs, depths, intrinsics = self.transform([tgt_img] + ref_imgs, [tgt_depth] + ref_depths, np.copy(sample['intrinsics']),self.height,self.width)
            tgt_img = imgs[0]     
            tgt_depth = depths[0]
            ref_imgs = imgs[1:]
            ref_depths = depths[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs,ref_poses, intrinsics, tgt_depth,ref_depths  


    def __len__(self):
        return len(self.samples)

