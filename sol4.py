# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy as si
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass,map_coordinates
import shutil
from imageio import imwrite

import sol4_utils

def harris_corner_detector(im):
  """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.  
  """
  k=0.04
  kernel_size=3
  I_x,I_y=conv_der(im)
  I_x2,I_y2,I_yx=harris_mat(I_x, I_y,kernel_size)
  R=find_R(I_x2,I_y2,I_yx,k)
  corners=non_maximum_suppression(R)
  corne_positions=np.nonzero(corners)
  corne_position=np.array([[corne_positions[1][i],corne_positions[0][i]] for\
          i in range(len(corne_positions[0]))]).reshape(len(corne_positions[0]),2)
  return corne_position

def harris_mat(I_x,I_y,kernel_size):
  """
  genreate the harris matrix
  :param I_x:
  :param I_y:
  :param kernel_size:
  :return: harris matrix elements
  """
  I_x2=sol4_utils.blur_spatial(I_x**2, kernel_size)
  I_y2 = sol4_utils.blur_spatial(I_y**2, kernel_size)
  I_yx=sol4_utils.blur_spatial(I_y*I_x, kernel_size)
  return I_x2,I_y2,I_yx

def find_R(I_x2,I_y2,I_yx,k):
  """
  calculate Response image
  :param I_x2:
  :param I_y2:
  :param I_yx:
  :param k:
  :return: R
  """
  det=I_x2*I_y2-I_yx**2
  trace=I_x2+I_y2
  return det-k*trace**2

def conv_der(im):
    """
    calculate image deriviativs
    :param im:
    :return:
    """
    dx_op = np.array([1, 0, -1]).reshape((1, 3))
    dy_op = np.array([1, 0, -1]).reshape((3, 1))
    dx = si.signal.convolve2d(im, dx_op, 'same','symm')
    dy = si.signal.convolve2d(im, dy_op, 'same','symm')
    return dx,dy

def sample_descriptor(im, pos, desc_rad):
  """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image.
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.   
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
  N,K=[len(pos),1+2*desc_rad]
  descriptors = np.zeros((N, K, K), dtype=np.float64)
  for i in range(N):
    x, y = np.indices((K, K)).astype(np.float64)
    x -= desc_rad - pos[i, 1]
    y -= desc_rad - pos[i, 0]
    descriptors[i,:,:] = map_coordinates(im, [x.flatten(), y.flatten()],order=1, prefilter=False).reshape((K, K))
    descriptors[i, :, :] =normalize(descriptors[i, :, :])
  return descriptors


def normalize(window):
  """
  normlize the window
  :param window:
  :return: normlize window
  """
  mean = np.mean(window)
  norm= np.linalg.norm(window - mean)
  if norm>0:
    norm_wind = (window - mean) /norm
  else:
    norm_wind=np.zeros(np.shape(window))
  return norm_wind


def find_features(pyr):
  """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image. 
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
  cordinates = spread_out_corners(pyr[0], 7, 7, 5)
  pos=cordinates*(2**-2)
  desc_rad=3
  descriptors=sample_descriptor(pyr[2], pos, desc_rad)
  return cordinates,descriptors





def match_features(desc1, desc2, min_score):
  """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
  k_biggest=2
  k=np.shape(desc1[1])[0]
  sjk_mat=np.dot(desc1.reshape(-1,k**2),desc2.reshape(-1,k**2).T)#bulid the matrix
  min_score_demand = sjk_mat > min_score #a bool matrix(True>min_score else Fales
  N1,N2=[desc1.shape[0], desc2.shape[0]]#dims of sjk matrix
  matched_array=np.zeros((N1,N2))#array that will contain all good values
  for row in range(N1):#loop every row calculate the 2 max values
    max_in = np.argpartition(sjk_mat[row, :],-k_biggest)[-k_biggest:]
    matched_array[row,max_in]+=1#add 1 to the matrix element

  for col in range(N2):#loop every col calculate the 2 max values
    max_in = np.argpartition(sjk_mat[:,col],-k_biggest)[-k_biggest:]
    matched_array[max_in,col] += 1

  matched_array=matched_array ==k_biggest #the element on the zero matrix
  # that equale 2,will turn to true(that means that they passed 2 of the
  # requirments
  matched_array=matched_array * min_score_demand#multiply to bool matrix
  # will give as the min score requirment

  return np.where(matched_array)


def apply_homography(pos1, H12):
  """
  Apply homography to inhomogenous points.
  :param pos1: An array with shapply_homography(pos1, H12)ape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
  N=np.shape(pos1)[0]
  d3_pos1=np.ones((N,3))
  d3_pos1[:,0:2]=pos1 #build the homogenious coordintaes
  pos_vectors = d3_pos1.T #transpose to perfurme multiplycation
  tranformd= np.dot(H12, pos_vectors).T.reshape(d3_pos1.shape) #the new homoginioud cords
  inv=(1/tranformd[:,2]).reshape(np.shape(tranformd)[0],1)
  new_cords=(tranformd[:,0:2]*inv).reshape(pos1.shape)
  return new_cords

def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
  """
  Computes homography between two sets of points using RANSAC.
  :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param pos2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
  shuffled =np.random.choice(np.arange(points1.shape[0]),size=num_iter+1,replace=True)
  best_inliers=np.array([])
  if translation_only==False:
    for i in range(num_iter):
      point1=np.array([points1[shuffled[i]],points1[shuffled[i+1]]]).reshape(2,2)
      point2 = np.array([points2[shuffled[i]],points2[shuffled[i+1]]]).reshape(2,2)
      curr_inliers = find_inliers(inlier_tol, point1, point2, points1,points2,translation_only)
      if curr_inliers.size > best_inliers.size:
          best_inliers = curr_inliers
  else:
    for i in range(num_iter):
      point1 = np.array([points1[shuffled[i]]]).reshape(1,2)
      point2 = np.array([points2[shuffled[i]]]).reshape(1,2)
      curr_inliers = find_inliers(inlier_tol, point1, point2, points1,points2, translation_only)
      if curr_inliers.size > best_inliers.size:
        best_inliers = curr_inliers
  if len(best_inliers)!=0:
    Homog = estimate_rigid_transform(points1[best_inliers,:],points2[best_inliers,:],translation_only)
    return  [Homog,best_inliers]
  return [[],best_inliers]

def find_inliers(inlier_tol, point1, point2, points1, points2,translation_only):
  """
  find the inliers from a set of points
  :param inlier_tol:
  :param point1:
  :param point2:
  :param points1:
  :param points2:
  :param translation_only:
  :return: inliers array
  """
  H12 = estimate_rigid_transform(point1, point2, translation_only)
  point2_tr = apply_homography(points1, H12)
  E = (np.linalg.norm(point2_tr - points2, axis=1))
  curr_inlaiers = np.where(E < inlier_tol)
  return curr_inlaiers[0]


def lines(pos1, pos2, color, line_width):
  """
  plot the lines between dots
  :param pos1: points of im 1
  :param pos2:  points of im 2
  :param color: color of the line
  :param line_width: width of the line
  """
  for i in range(pos1.shape[0]):
    po1,po2= [[pos1[i, 0], pos2[i, 0]],[pos1[i, 1], pos2[i, 1]]]
    plt.plot(po1,po2, color + "-", lw=line_width)


def display_matches(im1, im2, points1, points2, inliers):
  """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
  back_round = np.hstack((im1, im2))
  points2_moved = points2.copy()
  points2_moved[:, 0] += im1.shape[1]
  full_points = np.vstack((points1, points2_moved))
  point1_inliers = points1[inliers, :]
  point2_inliers = points2_moved[inliers, :]
  # plotting
  plt.plot(full_points[:, 0], full_points[:, 1], 'ro', markersize=1.5)
  plt.imshow(back_round, cmap='gray')  # plot the combined image
  lines(points1, points2_moved, "b", 0.4)  # plot the lines of out liners
  lines(point1_inliers, point2_inliers, "y", 0.6)  # plot the inliers lines
  plt.show()

def accumulate_homographies(H_succesive, m):
  """
  Convert a list of succesive homographies to a 
  list of homographies to a common reference frame.
  :param H_successive: A list of M-1 3x3 homography 
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to 
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices, 
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """ 
  H2m=[]
  for i in range(len(H_succesive)+1):
    if i<m:
       H2m.append(smaller_than_m(H_succesive,m,i))
    elif i>m:
      H2m.append(bigger_than_m(H_succesive, m, i))
    else:
      H2m.append(np.eye(3))
  return H2m

def bigger_than_m(H_succesive, m, i):
  Homog =np.eye(3)
  for j in range(m,i):
      Homog=np.matmul(Homog,np.linalg.inv(H_succesive[j]))
  return normalize_H(Homog)

def smaller_than_m(H_succesive,m,i):
  Homog = np.eye(3)
  for j in range(m-1,i-1,-1):
      Homog=np.matmul(Homog,H_succesive[j])
  return normalize_H(Homog)

def normalize_H(H):
  H=H.astype('float64')
  H /= H[2,2]
  return H


def compute_bounding_box(homography, w, h):
  """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """
  points=np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
  new_points=np.round(apply_homography(points, homography)).astype('int')
  x_sort = np.sort(new_points[:,0], axis=0)
  y_sort = np.sort(new_points[:,1], axis=0)
  top_left_corn=[x_sort[0],y_sort[0]]
  bottom_rghit_corn=[x_sort[-1],y_sort[-1]]
  return np.array([top_left_corn,bottom_rghit_corn])



def warp_channel(image, homography):
  """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
  hight,width=np.shape(image)
  [[x_min,y_min],[x_max,y_max]]=compute_bounding_box(homography,width,hight)
  x_cords,y_cords=[np.arange(x_min,x_max+1),np.arange(y_min,y_max+1)]
  Xi,Yi=np.meshgrid(x_cords,y_cords)
  positions = np.vstack([Xi.ravel(), Yi.ravel()]).T
  new_cords=apply_homography(positions,np.linalg.inv(homography)).reshape(Xi.shape[0],Xi.shape[1],2)
  Xi_t,Yi_t=new_cords[:,:,0],new_cords[:,:,1]
  new_image=map_coordinates(image, [Yi_t, Xi_t], order=1, prefilter=False)
  return new_image
def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret


def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) & 
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    points_and_descriptors = []
    for file in self.files:
      image = sol4_utils.read_image(file, 1)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]


  def generate_panoramic_images(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]


  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)


    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()


