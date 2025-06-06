bb = breakpoint
import torch
import numpy as np
from utils.utils import to_tensor, to_numpy
import open3d as o3d
import rerun as rr

OPENCV2OPENGL =  (1,-1,-1,1)

def pts_to_opengl(pts):
    return pts*OPENCV2OPENGL[:3]

def save_pointmaps(data,path='debug',view=False,color='novelty',frustrum_scale=20):
    # debug function to save points to a ply file
    import open3d as o3d
    pointmaps = data['pointmaps']
    B = pointmaps.shape[0]
    W, H = pointmaps.shape[-3:-1]
    n_cams = data['c2ws'].shape[1]
    geometries = []
    for b in range(B):
        geometry_b = []
        points = torch.cat([p.flatten(start_dim=0,end_dim=1) for p in pointmaps[b]],dim=0)
        if view:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(to_numpy(points))
            if color == 'novelty':
                colors = torch.ones_like(points)
                pts_p_cam = W*H
                # make all novel points red
                colors[pts_p_cam:,1:]*=0.1

                # make all points from first camera blue
                colors[:pts_p_cam,0]*=0.1
                colors[:pts_p_cam,2]*=0.1
                colors*=255.0
            
            else:
                colors = torch.cat([p.flatten(start_dim=0,end_dim=1) for p in data['imgs'][b]],dim=0)
            pcd.colors = o3d.utility.Vector3dVector(to_numpy(colors)/255.0)
            geometry_b.append(pcd)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=10, origin=[0,0,0])
            geometry_b.append(origin)
            for i in range(n_cams):
                K = data['Ks'][b,i].cpu().numpy()
                K = o3d.camera.PinholeCameraIntrinsic(W,H,K)
                P = data['c2ws'][b,i].cpu().numpy()
                cam_frame = o3d.geometry.LineSet.create_camera_visualization(intrinsic=K,extrinsic=P,scale=frustrum_scale)
                geometry_b.append(cam_frame)
            o3d.visualization.draw_geometries(geometry_b)

            # add point at the origin
            o3d.io.write_point_cloud(f"{path}_{b}.ply", pcd)
            breakpoint()
        geometries.append(geometry_b)
    return geometries

def just_load_viz(pred_dict,gt_dict,batch,name='just_load_viz',addr='localhost:9000',fused_meshes=None,n_points=None):
    rr.init(name)
    rr.connect(addr)
    rr.set_time_seconds("stable_time", 0)

    context_views = batch['input_cams']['pointmaps']
    context_rgbs = batch['input_cams']['imgs']
    gt_pred_views = gt_dict['pointmaps']
    pred_views = pred_dict['pointmaps']
    
    # FIX this weird shape
    pred_masks = batch['new_cams']['valid_masks']
    context_masks = batch['input_cams']['valid_masks']

    B = batch['new_cams']['pointmaps'].shape[0]
    W,H = context_views.shape[-3:-1]
    n_pred_cams = pred_views.shape[1]

    for b in range(B):
        rr.set_time_seconds("stable_time", b)
        # Set world transform to identity (normal origin)
        rr.log("world", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)))
        ## show context views
        context_rgb = to_numpy(context_rgbs[b])
        
        for i in range(n_pred_cams):
            if 'conf_pointmaps' in pred_dict:
                conf_pts = pred_dict['conf_pointmaps'][b,i]
                
                #print(f"view {i} mean conf: {mean_conf}, std conf: {std_conf}")
                conf_pts = (conf_pts - conf_pts.min())/(conf_pts.max() - conf_pts.min())
                conf_pts = to_numpy(conf_pts)
                rr.log(f"view_{i}/pred_conf", rr.Image(conf_pts))
            if pred_masks[b,i].sum() == 0:
                continue
            if gt_pred_views is not None:
                gt_pred_pts = gt_pred_views[b,i][pred_masks[b,i]]
                gt_pred_pts = to_numpy(gt_pred_pts)
            else:
                gt_pred_pts = None 

            # red is color for gt points
            if gt_pred_pts is not None:
                color = np.array([1,0,0])
                colors = np.ones_like(gt_pred_pts)
                colors[:,0] = color[0]
                colors[:,1] = color[1]
                colors[:,2] = color[2]
                rr.log(
                    f"world/new_views_gt/view_{i}", rr.Points3D(gt_pred_pts,colors=colors)
                )
            # green is color for pred points
            pred_pts = pred_views[b,i][pred_masks[b,i]]
            pred_pts = to_numpy(pred_pts)

            depth = pred_views[b,i][:,:,2]
            depth -= depth[pred_masks[b,i]].min()
            depth[~pred_masks[b,i]] = 0
            depth /= depth.max()
            depth = to_numpy(depth)
            rr.log(f"world/new_views_pred/view_{i}/image", rr.Image(depth))

            if 'classifier' in pred_dict:
                classifier = (pred_dict['classifier'][b,i] > 0.0).float() # this is assuming the classifier is a sigmoid output
                classifier = to_numpy(classifier)
                rr.log(f"view_{i}/pred_mask", rr.Image(classifier))

            color = np.array([0,1,0])
            colors = np.ones_like(pred_pts)
            colors[:,0] = color[0]
            colors[:,1] = color[1]
            colors[:,2] = color[2]
            if n_points is None:
                rr.log(
                    f"world/new_views_pred/view_{i}/pred_points", rr.Points3D(pred_pts,colors=colors)
                )
            else:
                # randomly sample n_points from pred_pts
                n_points = min(n_points, pred_pts.shape[0])
                inds = np.random.choice(pred_pts.shape[0], n_points, replace=False)
                rr.log(
                    f"world/new_views_pred/view_{i}/pred_points", rr.Points3D(pred_pts[inds],colors=colors[inds])
                )

            K = batch['new_cams']['Ks'][b,i].cpu().numpy()
            P = batch['new_cams']['c2ws'][b,i].cpu().numpy()
            P = np.linalg.inv(P)
            rr.log(f"world/new_views_pred/view_{i}", rr.Transform3D(translation=P[:3,3], mat3x3=P[:3,:3], from_parent=True))
            
            rr.log(f"world/new_views_gt/view_{i}", rr.Transform3D(translation=P[:3,3], mat3x3=P[:3,:3], from_parent=True))

            if 'classifier' in pred_dict:
                classifier = gt_dict['valid_masks'][b,i].float()
                classifier = to_numpy(classifier)
                rr.log(f"view_{i}/gt_mask", rr.Image(classifier))

            rr.log(
                f"world/new_views_pred/view_{i}/image",
                rr.Pinhole(
                    resolution=[H, W],
                    focal_length=[K[0,0], K[1,1]],
                    principal_point=[K[0,2], K[1,2]],
                ),
            )

            rr.log(f"world/new_views_pred/view_{i}/image", rr.Image(to_numpy(pred_masks[b,i].float())))
        n_input_cams = context_masks.shape[1]

        for i in range(n_input_cams):
            context_pts = context_views[b][i][context_masks[b][i]]
            context_pts = to_numpy(context_pts)
            context_pts_rgb = context_rgbs[b][i][context_masks[b][i]]
            context_pts_rgb = to_numpy(context_pts_rgb)
            
            # depth imgs 
            #context_depths = batch['input_cams']['depths'][b][i]
            #context_depths  = (context_depths / context_depths.max() * 255.0).clamp(0,255)
            #context_depths = to_numpy(context_depths).astype(np.uint8)
            rr.log(
                f"world/context_views/view_{i}_points", rr.Points3D(context_pts,colors=(context_pts_rgb/255.0))
            )
            
            K = batch['input_cams']['Ks'][b,i].cpu().numpy()
            P = batch['input_cams']['c2ws'][b,i].cpu().numpy()
            P = np.linalg.inv(P)
            rr.log(f"world/context_views/view_{i}", rr.Transform3D(translation=P[:3,3], mat3x3=P[:3,:3], from_parent=True))

            rr.log(
                f"world/context_views/view_{i}/image",
                rr.Pinhole(
                    resolution=[H, W],
                    focal_length=[K[0,0], K[1,1]],
                    principal_point=[K[0,2], K[1,2]],
                        ),
                   )
            context_rgb_i = context_rgb[i]
            rr.log(
                f"world/context_views/view_{i}/image", rr.Image(context_rgb_i)
            )

            rr.log(
                f"world/context_camera_{i}/mask", rr.Image(to_numpy(context_masks[b,i].float()))
            )
        if fused_meshes is not None:
            rr.log(f"world/fused_mesh", rr.Mesh3D(vertex_positions=fused_meshes[b]['verts'], vertex_normals=fused_meshes[b]['norms'], vertex_colors=fused_meshes[b]['colors'], triangle_indices=fused_meshes[b]['faces']))

         