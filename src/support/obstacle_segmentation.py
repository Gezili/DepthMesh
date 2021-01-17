import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

#Define camera parameters
omni_cam_0 = [0.000,0.004,0.056,0.002,0.001,-0.006,1.000]
omni_cam_1 = [-0.001,0.127,0.054,0.005,0.002,-0.002,1.000]

omni_cam_0_intrinsics_and_dist = [482.047,485.211,373.237,211.02,-0.332506,0.154213,-9.5973e-05,-0.000236179,-0.0416498]

omni_cam_1_intrinsics_and_dist = [479.429,482.666,367.111,230.626,-0.334792,0.161382,4.29188e-05,-0.000324466,-0.0476611]

class Camera:

    '''
    A camera object.

    Holds all parameters required to calibrate and rectify
    cameras with rotation and translation wrt omni frame,
    distortion parameters and the projection matrix
    '''

    def __init__(self, rot_and_trans, intrinsics_and_dist):

        #Get translation from translation parameters
        #Note we only ever work with relative translation
        #and rotation, so at no point do we need to convert
        #to the world frame
        self.trans = np.array(rot_and_trans[:3])
        self.projection_matrix = np.eye(4)
        self.image_shape = (480, 752)

        #Get rotation matrix from quaterion representation
        self.rot = R.from_quat(rot_and_trans[3:])
        self.rot = self.rot.as_matrix()

        self.projection_matrix[:3,:3] = self.rot
        self.projection_matrix[:3,3] = self.trans

        #Generate camera intrinsics matrix from intrinsic parameters
        self.intrinsics_matrix = np.array([
            [intrinsics_and_dist[0], 0, intrinsics_and_dist[2]],
            [0, intrinsics_and_dist[1], intrinsics_and_dist[3]],
            [0, 0, 1]
        ])

        #Get distortion

        self.distortion = np.array(intrinsics_and_dist[4:])

#We will be working with cams 0 and 1 for stereo images
cam_1 = Camera(omni_cam_0, omni_cam_0_intrinsics_and_dist)
cam_2 = Camera(omni_cam_1, omni_cam_1_intrinsics_and_dist)

def load_frame(frame, input_dir):

    '''
    Load all images from all the omnidirectional cameras
    for a given frame.
    '''

    file_path = input_dir + '/run4_base_hr/'
    frame_prefix = 'frame' + str(frame).zfill(6)

    images = []

    for i in range(10):
        dir = file_path + f'omni_image{i}'
        frames = os.listdir(dir)

        pic_to_load = [frame for frame in os.listdir(dir) if frame_prefix in frame].pop()

        image = cv2.imread(dir + f'/{pic_to_load}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    return images

def generate_depthmap(frame, input_dir, output_dir, plot=False):

    '''
    Generate the depthmap for a pair of stereo images for a given
    frame.
    '''

    cam1 = 0
    cam2 = 1

    images = load_frame(frame, input_dir)
    image_l = images[cam1]
    image_r = images[cam2]

    #We need to rectify these cameras first - calculate
    #rectification parameters

    R1, R2, P1, P2, Q, ROI1, ROI2 = cv2.stereoRectify(
        cam_1.intrinsics_matrix,
        cam_1.distortion,
        cam_2.intrinsics_matrix,
        cam_2.distortion,
        (480, 752),
        np.linalg.inv(cam_1.rot)@cam_2.rot,
        cam_2.trans - cam_1.trans
    )

    #Find maps to rectify to

    map_1_i, map_1_f = cv2.initUndistortRectifyMap(
        cam_1.intrinsics_matrix,
        cam_1.distortion,
        R1,
        P1,
        (752, 480),
        cv2.CV_32FC1
    )

    map_2_i, map_2_f = cv2.initUndistortRectifyMap(
        cam_2.intrinsics_matrix,
        cam_2.distortion,
        R2,
        P2,
        (752, 480),
        cv2.CV_32FC1
    )

    #Apply rectification

    image_l = cv2.remap(image_l, map_1_i, map_1_f, cv2.INTER_LINEAR)
    image_r = cv2.remap(image_r, map_2_i, map_2_f, cv2.INTER_LINEAR)

    image_l_color= image_l

    #Note that these cameras are top and bottom instead of
    #side by side, so to compute disparity we need to rotate
    #the two images

    image_l = cv2.rotate(image_l, cv2.ROTATE_90_CLOCKWISE)
    image_r = cv2.rotate(image_r, cv2.ROTATE_90_CLOCKWISE)

    #Convert to grayscale

    image_l = cv2.cvtColor(image_l, cv2.COLOR_RGB2GRAY)
    image_r = cv2.cvtColor(image_r, cv2.COLOR_RGB2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=50,
        blockSize=20,
        speckleWindowSize=200,
        speckleRange=1,
        mode=4
    )

    disparity = stereo.compute(image_r, image_l)

    if plot:

        #Save the figures

        plt.subplot(121)

        plt.imshow(image_l_color)

        plt.xticks([])
        plt.yticks([])
        plt.title('Original Image - Rectified')

        plt.subplot(122)
        disparity_plot = disparity
        disparity_plot[:,210:] = -1
        disparity_plot = disparity_plot - disparity_plot.min()
        disparity_plot = disparity_plot/disparity_plot.max()
        disparity_plot = np.rot90(disparity_plot)

        plt.xticks([])
        plt.yticks([])
        plt.title('Depth Map')

        plt.imshow(disparity_plot, 'gray')
        plt.savefig(output_dir + f'/disparity_{frame}.png')
        plt.close()

    disparity = disparity[:,:210]
    mask = ~(disparity == -16)
    disparity += 16*6*mask
    disparity = disparity.astype('float32')

    #Transform disparity map into point cloud formulation

    points = cv2.reprojectImageTo3D(disparity, Q)

    return image_l_color, points

def generate_segmentation_mask(
    frame, input_dir, output_dir,
    plot=False, plot_meshes=False
):

    '''
    Here we generate the segmentation masks for the image,
    using point cloud data.
    '''

    #Get depthmap
    rect_img, orig_pc = generate_depthmap(
        input_dir=input_dir,
        output_dir=output_dir,
        frame=frame,
        plot=True
    )

    orig_pc = orig_pc.reshape(-1,3)

    #Remove all the points that were failed to be detected by
    #the depth algorithm

    pc = orig_pc[orig_pc[:,2] < 0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    #Here we downsample the pointcloud - estimating surface normals
    #takes a long time and therefore we do not want every single
    #point.
    pcd = pcd.uniform_down_sample(5)

    if plot_meshes:
        o3d.visualization.draw_geometries([pcd])

    #Estimate point normals by fitting a surface to points
    #within a certain neighborhood of that point

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    #Do to the way normals are defined, make normals consistent
    #and orient them to face upwards

    pcd.orient_normals_consistent_tangent_plane(180)

    #Use the ball pivoting algorithm in order to generate
    #a mesh from the pointcloud and their normals

    radii = [0.005, 0.01, 0.02, 0.04, 0.1]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))

    #Here we compute the normals of the triangles of the mesh,
    #and also smooth the image.

    mesh.compute_triangle_normals()
    mesh.filter_smooth_laplacian()

    normals = np.asarray(mesh.triangle_normals)
    triangles = np.asarray(mesh.triangles)

    #Compute the median slope of the scene without needing
    #to go into world frame
    dist_from_std = np.linalg.norm((normals - np.median(normals, axis=0))/normals.std(axis=0), axis=1)

    #Remove triangles with normals that are more than two standard
    #deviations away from the median of the frame
    triangles = triangles[dist_from_std < 2]
    normals = normals[dist_from_std < 2]

    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    #Find connected regions of meshes
    regions = mesh.cluster_connected_triangles()[0]

    #Assume that the largest connected region is the current location
    #of the rover. Set this to the traversable space

    largest_connected_region = np.bincount(regions).argmax()

    normals = np.asarray(mesh.triangle_normals)
    triangles = np.asarray(mesh.triangles)

    triangles = triangles[regions == largest_connected_region]
    normals = normals[regions == largest_connected_region]

    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_normals = o3d.utility.Vector3dVector(normals)

    points = np.asarray(mesh.vertices)

    #Take the vertices of the triangles associated with the
    #largest connected component

    tri_verts = list(set(np.asarray(mesh.triangles).flatten()))
    points = points[tri_verts]

    print(f'Pruned mesh with {points.shape[0]} triangles.')

    if plot_meshes:
        o3d.visualization.draw_geometries([pcd, mesh])

    #Calculate rectification here in order to reproject pointcloud
    #back into image space

    rec = cv2.stereoRectify(
        cam_1.intrinsics_matrix,
        cam_1.distortion,
        cam_2.intrinsics_matrix,
        cam_2.distortion,
        cam_1.image_shape,
        np.linalg.inv(cam_1.rot)@cam_2.rot,
        cam_2.trans - cam_1.trans
    )

    Q = rec[-3]

    f = Q[2][3]
    a = Q[3][2]
    b = Q[3][3]
    Cx = -Q[0][3]
    Cy = -Q[1][3]

    X = points[:,0]
    Y = points[:,1]
    Z = points[:,2]

    d = (f - Z*b)/(Z*a)
    u = X*(d*a + b) + Cx
    v = Y*(d*a + b) + Cy

    #Points belonging to the traversable space
    data = np.vstack([u,v])
    mask = np.zeros(rect_img.shape[:2])

    s_x = rect_img.shape[0]
    s_y = rect_img.shape[1]
    dp = 1

    #For each pixel of the traversable space, fit
    #mark a window around it as traversable. Window
    #size is a parameter here

    for point in data.T:

        x = int(np.clip(point[0], 0, s_x-1))
        y = int(np.clip(point[1], 0, s_y-1))

        mask[x - dp: x + dp, y - dp: y + dp] = 1

    #Blur this mask, then apply thresholding in order to determine
    #the traversable and untraversable masks

    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    thresh = 0.1

    mask[mask >= thresh] = 1
    mask[mask < thresh] = 0
    mask = np.flipud(mask)
    mask = 1 - mask

    if plot:

        #Plot the untraversable mask overlaid on top
        #of the image
        colour_mask = np.zeros(list(rect_img.shape))
        colour_mask[...,0] = mask

        seg_img = rect_img/255*0.8 + colour_mask*0.2
        plt.imshow(seg_img)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(output_dir + f'/obstacle_mask_{frame}.png')
        plt.close()

    return mask

if __name__ == '__main__':

    generate_segmentation_mask(1400, plot=True, plot_meshes=True)