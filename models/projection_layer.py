from mxnet import nd


def set_id_grid(depth):
    b, h, w = depth.shape
    i_range = nd.arange(0, h, ctx=depth.context, dtype=depth.dtype).reshape((1, h, 1)).broadcast_to((1, h, w))
    j_range = nd.arange(0, w, ctx=depth.context, dtype=depth.dtype).reshape((1, 1, w)).broadcast_to((1, h, w))
    ones = nd.ones((1,h,w), ctx=depth.context, dtype=depth.dtype)
    pixel_coords = nd.stack(*[j_range, i_range, ones], axis=1)  # [1, 3, H, W]
    return pixel_coords

def check_sizes(input, input_name, expected):
    condition = [input.ndim == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.shape[i] == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.shape))

def pixel2cam(depth, intrinsics_inv, pixel_coords):
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.shape
    if (pixel_coords is None) or pixel_coords.shape[2] < h or pixel_coords.shape[3] < w:
        pixel_coords = set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].broadcast_to((b, 3, h, w)).reshape((b, 3, -1))  # [B, 3, H*W]
    cam_coords = nd.linalg.gemm2(intrinsics_inv, current_pixel_coords).reshape((b, 3, h, w))
    depth = depth.reshape((b, 1, h, w))
    return cam_coords * depth

def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.shape
    cam_coords_flat = cam_coords.reshape((b, 3, h * w))  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = nd.linalg.gemm2(proj_c2p_rot, cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2] # .clip(a_min=1e-3)
    Z = nd.maximum(Z, 1e-3)

    X_norm = 2 * (X / Z) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
    X_mask = ((X_norm > 1) + (X_norm < -1))
    Y_mask = ((Y_norm > 1) + (Y_norm < -1))

    if padding_mode == 'zeros':
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_norm[Y_mask] = 2
    mask = ((X_norm > 1) + (X_norm < -1) + (Y_norm < -1) + (Y_norm > 1))
    mask = mask.reshape((b, 1, h, w)).broadcast_to((b, 3, h, w)) # [B, 3, H, W]
    mask = nd.minimum(mask, 1)

    pixel_coords = nd.stack(*[X_norm, Y_norm], axis=2)  # [B, H*W, 2]
    return pixel_coords.reshape((b, h, w, 2)), mask

def inverse_warp(img, depth, pose_mat, intrinsics, padding_mode='border'):
    """
    Inverse warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose_mat: 6DoF pose parameters from target to source as 4x4 matrix -- [B, 4, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    depth = depth[:,0,:,:]
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(intrinsics, 'intrinsics', '133')
    batch_size, _, img_height, img_width = img.shape

    pixel_coords = set_id_grid(depth)
    intrinsics = intrinsics.broadcast_to((img.shape[0], intrinsics.shape[1], intrinsics.shape[2]))

    cam_coords = pixel2cam(depth, nd.linalg.inverse(intrinsics), pixel_coords)  # [B,3,H,W]
    # Get projection matrix for the camera frame to source pixel frame
    proj_cam_to_src_pixel = nd.linalg.gemm2(intrinsics, pose_mat[:,:3,:])  # [B, 3, 4]


    src_pixel_coords, mask = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]
    src_pixel_coords = nd.transpose(src_pixel_coords, (0, 3, 1, 2))
    src_pixel_coords = nd.clip(src_pixel_coords, a_min=-1.0, a_max=1.0)

    # projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
    projected_img = nd.BilinearSampler(data=img, grid=src_pixel_coords)
    return projected_img, src_pixel_coords, mask