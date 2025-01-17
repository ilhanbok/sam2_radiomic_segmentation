# Take existing SAM / Mirror pipeline and use SAM2 video parsing capabilties
# Dec 17, 2024 | Ilhan Bok, Patrick Belton

import numpy as np
import nibabel as nib
from scipy.ndimage import rotate, shift
from skimage.measure import regionprops, label
import os

import pickle

def compute_asymmetry_index(slice,i):
    F = np.flipud(slice)
    asymmetry_index_matrix = np.abs(slice - F)
    return asymmetry_index_matrix

    I = slice

    # Get the dimensions of the image
    rows, columns = I.shape

    # Binary image creation (assuming already binary or thresholded)
    binaryImage = I > 0  # Convert to boolean for binary image
    # Label the image and measure properties
    labeledImage = label(binaryImage)
    props = regionprops(labeledImage)

    # Check if any region is detected
    if not props:
        return(np.zeros(I.shape))
    print(props[0].orientation)
    #quit()
    centroid = props[0].centroid
    xCentroid, yCentroid = centroid

    # Middle of the image
    middlex = columns / 2
    middley = rows / 2

    # Translate the image to the center
    deltax = 0#middlex - xCentroid
    deltay = 0#middley - yCentroid
    translatedImage = shift(I, [deltay, deltax])

    theta = -props[0].orientation
    # Rotate the image (adjust angle if needed) # This is the default which should be adjusted in the future
    rotatedImage = rotate(translatedImage, theta, reshape=False)

    # Compute the asymmetry by reflecting the image and calculating absolute differences
    F = np.flipud(rotatedImage)
    asymmetry_index_matrix = np.abs(rotatedImage - F)

    # Extra step: transform the image back


    # Rotate the image (adjust angle if needed) # This is the default which should be adjusted in the future
    rotated_asymmetry_index_matrix = rotate(asymmetry_index_matrix, -theta, reshape=False)
    # Translate the image to the center
    translated_rotated_asymmetry_index_matrix = shift(rotated_asymmetry_index_matrix, [-deltay, -deltax])

    return translated_rotated_asymmetry_index_matrix

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

import numpy as np

def fill_bounded_regions(binary_array):
    """
    Fills all bounded regions of 0s in a binary 3D NumPy array with 1s.

    Parameters:
        binary_array (np.ndarray): A binary 3D NumPy array (containing only 0s and 1s).

    Returns:
        np.ndarray: A binary 3D NumPy array with bounded regions filled.
    """
    if not np.array_equal(binary_array, binary_array.astype(bool)):
        raise ValueError("Input array must be binary (contain only 0s and 1s).")

    # Create a copy of the array to work on
    filled_array = binary_array.copy()

    # Get the shape of the array
    shape = binary_array.shape

    # Iterate until no more changes occur
    changed = True
    while changed:
        changed = False
        # Iterate over each element in the array
        for x in range(1, shape[0] - 1):
            for y in range(1, shape[1] - 1):
                if filled_array[x, y] == 0:
                    # Check if all directly adjacent entries are 1
                    if (filled_array[x - 1, y] == 1 and
                        filled_array[x + 1, y] == 1 and
                        filled_array[x, y - 1] == 1 and
                        filled_array[x, y + 1] == 1):
                        changed = True

    return filled_array


import nibabel as nib
import numpy as np
import torch

import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from scipy.ndimage import gaussian_filter

import SimpleITK as sitk

from PIL import *
import subprocess



# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"using device: {device}")


sam2_checkpoint = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Load the NIfTI file
nifti_path = "/Users/bok/Library/CloudStorage/Box-Box/Belton_Lab_Ilhan/QSM_Pipeline_Belton_Lab/qsm-proc-main/tgv/02_1070/Mag_Images.nii.gz"  # Replace with your NIfTI file path
# Load NIfTI image
nifti_image = nib.load(nifti_path)
image_data = nifti_image.get_fdata()

skull_strip = False
if skull_strip:
    output_path = "./temp_sks.nii.gz"
    command = ["bet", nifti_path, output_path, "-m", "-f", str(0.1)]
    subprocess.run(command, check=True)

    nifti_image = nib.load(output_path)
    image_data = nifti_image.get_fdata()


import matplotlib.pyplot as plt

pickleIt = True
if pickleIt:

    # Define a callback function for mouse clicks
    def on_click(event):
        # Store the coordinates of the click
        if event.inaxes:
            coordinates.append((event.xdata, event.ydata))
            print(f"Mouse clicked at: {event.xdata}, {event.ydata}")

            # Optionally, plot the point where clicked
            plt.plot(event.xdata, event.ydata, 'ro')
            plt.draw()
    slices = [91,146,76] #(0,1,2)
    # [58,82,153] #07_1073 (QSM)
    # [59,51,129] #05_1059 (QSM)
    # [34,89,140] #01_1007 (GRE)
    # [64,127,185] #01_1007 (MP_RAGE)
    # [47,144,110] #01_1011 (GRE)
    # [79,124,197] #05_1003 (MP_RAGE)
    for n in range(2,3):
        # Load the NIfTI file
        output_dir = "output_jpeg_stack_temp_1070_2_a"
        subprocess.run(['rm','-rf',output_dir])
        os.makedirs(output_dir, exist_ok=True)

        # Load the image
        img = nib.load(nifti_path)
        data = image_data
        # Normalize data to 0-255 for saving as JPEG
        data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
        data = data.astype(np.uint8)
        frame_names = []
        # Iterate through slices and save as JPEG
        for i in range(data.shape[n]):  # Assuming 3D data
            if n == 0:
                slice_data = data[i, :, :]
            if n == 1:
                slice_data = data[:, i, :]
            if n == 2:
                slice_data = data[:, :, i]

            # Restore orientation
            slice_data = (slice_data)

            img = Image.fromarray(slice_data)
            img.save(os.path.join(output_dir, f"{i:04d}.jpg"))
            frame_names.append(f"{i:04d}.jpg")

        print(f"JPEG stack saved to {output_dir}")

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
        inference_state = video_predictor.init_state(video_path=output_dir)

        ann_frame_idx = slices[n]-1 # subtract 1 in order to match ITK-SNAP indices
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # List to store coordinates'

        gusta = False
        #frame_names = np.array(list(range(data.shape[2]))).astype(str)
        while gusta == False:
            coordinates = []
            fig, ax=plt.subplots()
            plt.imshow(Image.open(os.path.join(output_dir, frame_names[ann_frame_idx])),interpolation='none')
            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            g = input('Gusta? (Si/No)')
            confirmed = False
            while confirmed == False:
                try:
                    if g == 'Si':
                        yn = input('Please enter fg (1) or bg (0) list: ')
                        yn = yn.split(' ')
                        gusta = True
                        confirmed = True
                    else:
                        h = input('Are you sure you want to discard? (Si/No)')
                        if h == 'Si':
                            confirmed = True
                except:
                    print('Incorrect input')
                finally:
                    pass
            # Let's add a positive click at (x, y)
            points = np.array(coordinates, dtype=np.float32)
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array(yn, np.int32)

            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(Image.open(os.path.join(output_dir, frame_names[ann_frame_idx])))
            show_points(points, labels, plt.gca())
            show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

            plt.show()
            gusta = False
            g = input('Gusta? (Si/No)')
            confirmed = False
            while confirmed == False:
                if g == 'Si':
                    gusta = True
                    confirmed = True
                else:
                    h = input('Are you sure you want to discard? (Si/No)')
                    if h == 'Si':
                        confirmed = True

            coordinates = []
            fig, ax=plt.subplots()
            plt.imshow(Image.open(os.path.join(output_dir, frame_names[ann_frame_idx])))
            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show()
            g = input('Gusta? (Si/No)')
            confirmed = False
            while confirmed == False:
                try:
                    if g == 'Si':
                        yn = input('Please enter fg (1) or bg (0) list: ')
                        yn = yn.split(' ')
                        gusta = True
                        confirmed = True
                    else:
                        h = input('Are you sure you want to discard? (Si/No)')
                        if h == 'Si':
                            confirmed = True
                except:
                    print('Incorrect input')
                finally:
                    pass
            # Let's add a positive click at (x, y)
            points = np.array(coordinates, dtype=np.float32)
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array(yn, np.int32)

            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id+1,
                points=points,
                labels=labels,
            )

            # show the results on the current (interacted) frame
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(Image.open(os.path.join(output_dir, frame_names[ann_frame_idx])))
            show_points(points, labels, plt.gca())
            for i in range(0,2):
                show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[i])

            plt.show()
            gusta = False
            g = input('Gusta? (Si/No)')
            confirmed = False
            while confirmed == False:
                if g == 'Si':
                    gusta = True
                    confirmed = True
                else:
                    h = input('Are you sure you want to discard? (Si/No)')
                    if h == 'Si':
                        confirmed = True

        # Now do the whole video
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            print(out_frame_idx)
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        # Beta: now run the other way to fill the whole image
        # reverse video

        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state,reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def rot270(img):
        return np.rot90(np.rot90(np.rot90(img)))

    with open('01_1059_20241226.pkl','wb') as f:
        pickle.dump(video_segments, f)
else:
    n = 2
    data = image_data
    with open('01_1059_20241226.pkl','rb') as f:
        video_segments = pickle.load(f)

def check_overlapping_entries(arr1, arr2):

    # Find the number of entries in common by position

    # Initialize counter
    counter = 0
    # Iterate through both arrs element by element
    for i in range(arr1.shape[0]):  # Rows
        for j in range(arr1.shape[1]):  # Columns
            if (arr1[i][j] <= arr2[i][j]) and arr1[i][j] == 1:
                counter += 1
    filt = arr2[np.where(arr1 == 1)]
    if counter > 0 and len(filt) > 0:
        max = np.mean(filt)#[np.where(filt<254)])
    else:
        max = 0
    #print(counter)
    return (counter, max)

size = (192,256,96)
# (128,192,192) # QSM
# (192,128,192) # QSM
# (128,180,192) # GRE
# (176,240,256) # MP-RAGE
from scipy import ndimage
nifti_res = np.zeros(size)
maxes = []
for i in range(len(video_segments)):
    x = 0
    for out_obj_id, out_mask in video_segments[i].items():
        if x == 0:
            if n == 0:
                curr_mask_0_symmetrycheck = compute_asymmetry_index(data[i,:,:]).astype(np.uint8)
                nifti_res[i,:,:] = (nifti_res[i,:,:] + out_mask[0])*(curr_mask_0_symmetrycheck)
            if n == 1:
                curr_mask_0_symmetrycheck = compute_asymmetry_index(data[:,i,:]).astype(np.uint8)
                nifti_res[:,i,:] = (nifti_res[:,i,:] + out_mask[0])*(curr_mask_0_symmetrycheck)
            if n == 2:
                print(i)
                curr_mask_0_symmetrycheck = compute_asymmetry_index((data[:,:,i]),i).astype(np.uint8)
                #curr_mask_0_symmetrycheck[np.where(curr_mask_0_symmetrycheck > 0)] = 1
                t = 180
                #curr_mask_0_symmetrycheck[np.where(curr_mask_0_symmetrycheck < t)] = 0
                #curr_mask_0_symmetrycheck[np.where(curr_mask_0_symmetrycheck >= t)] = 1
                #curr_mask_0_symmetrycheck = fill_bounded_regions(curr_mask_0_symmetrycheck)
                #curr_mask_0_symmetrycheck = ndimage.binary_fill_holes(curr_mask_0_symmetrycheck).astype(int)

                nifti_res[:,:,i] = curr_mask_0_symmetrycheck# + np.multiply(out_mask[0],curr_mask_0_symmetrycheck)
                #if np.any(out_mask[0] == curr_mask_0_symmetrycheck):
                counter,max = check_overlapping_entries(out_mask[0],curr_mask_0_symmetrycheck)#np.multiply(data[:,:,i],curr_mask_0_symmetrycheck))
                maxes.append(max)
                # if i == 110:
                #     print(max)
                #     plt.imshow(curr_mask_0_symmetrycheck)
                #     plt.show()
        x += 1

print(maxes)
max_lesion_cont = np.zeros((len(video_segments)))
prevmask = np.zeros((192,256))
iprev = 0
currmax = 0
print('Ranking lesions while determining volume...')
lesion_dict = {}
# contains information on (dummy example)
# lesion num : 1
# mean intensity whole lesion : 210.42 a.u.
# total intensity : 288*210.42 = 60600.96 voxel a.u.
# total volume : 288 voxels
# voxel volume : 4.33 mm^3
lesion_num = 0
insideLesion = False
insideLesionPrev = False

# Compute all the final stats for the lesion dictionary
sx, sy, sz = nifti_image.header.get_zooms()
VV = sx*sy*sz
if n == 0:
    VA = sy*sz
if n == 1:
    VA = sx*sz
if n == 2:
    VA = sx*sy

for i in range(len(video_segments)):
    x = 0
    for out_obj_id, out_mask in video_segments[i].items():
        if x == 0:
            print(out_mask[0].shape)
            print(prevmask.shape)
            counter,_= check_overlapping_entries(out_mask[0],prevmask)
            #plt.imshow(prevmask)
            #plt.show()
            print(np.where(out_mask[0]==prevmask))
            max = maxes[i]
            insideLesionPrev = insideLesion
            if counter > 0:
                insideLesion = True
                if insideLesion == True and insideLesionPrev == False:
                    lesion_dict[lesion_num] = {
                        'meanIntensities' : [],
                        'totalMeanIntensity' : 0,
                        'sliceAreas' : [],
                        'sliceAreasVoxel' : [],
                        'totalVolume' : 0,
                        'totalVolumeVoxel' : 0
                    }
                print(i)
                print('yes')
                print(max)
                print(currmax)
                currmax += max
                lesion_dict[lesion_num]['meanIntensities'].append(max)
                lesion_dict[lesion_num]['sliceAreas'].append(np.sum(out_mask[0]))
            else:
                insideLesion = False
                if insideLesion == False and insideLesionPrev == True:
                    l = lesion_num
                    lesion_dict[l]['totalVolume'] = np.sum(lesion_dict[l]['sliceAreas'])
                    lesion_dict[l]['totalVolumeVoxel'] = np.multiply(lesion_dict[l]['totalVolume'],VV)
                    lesion_dict[l]['sliceAreasVoxel'] = np.multiply(lesion_dict[l]['sliceAreas'],VA)
                    lesion_dict[l]['totalMeanIntensity'] = np.sum(np.multiply(lesion_dict[l]['sliceAreasVoxel'],lesion_dict[l]['meanIntensities']))/lesion_dict[l]['totalVolumeVoxel']
                    print(lesion_dict[l]['totalMeanIntensity'])
                    max_lesion_cont[iprev:i] = lesion_dict[l]['totalMeanIntensity']
                    lesion_num += 1
                currmax = 0
                iprev = i
            prevmask = out_mask[0]
        x += 1

# Complete the whole loop by filling in the remaining items
max_lesion_cont[iprev:] = currmax
print(max_lesion_cont)

for i in range(len(lesion_dict)):
    print('Lesion #: {0}'.format(i))
    print('Total Volume (mm^3): {0}'.format(lesion_dict[i]['totalVolumeVoxel']))
    print('Mean Lesion Asymmetry Index: {0}'.format(lesion_dict[i]['totalMeanIntensity']))

nifti_res_SAMonly_lesionrank = np.zeros(size)
print('Applying rankings...')
for i in range(len(video_segments)):
    for out_obj_id, out_mask in video_segments[i].items():
        nifti_res_SAMonly_lesionrank[:,:,i] = nifti_res_SAMonly_lesionrank[:,:,i] + np.multiply(out_mask[0],max_lesion_cont[i])#curr_mask_0_symmetrycheck
print(np.where(nifti_res_SAMonly_lesionrank > 0))
nifti_mask = nib.Nifti1Image(nifti_res, affine=nib.load(nifti_path).affine)
nib.save(nifti_mask, 'output_video_02_1070_2_10152014_23u489123u+dir={0}_rot90_sym_rawdata_37489237_1_overlap190_countermode_2_skullstripped.nii.gz'.format(n))

#nifti_mask = nib.Nifti1Image(nifti_res_SAMonly, affine=nib.load(nifti_path).affine)
#nib.save(nifti_mask, 'output_video_01_1007_20140402_23u489123u+dir={0}_rot90_sym_rawdata_37489237_1Samfilt_12262024-2.nii.gz'.format(n))

nifti_mask = nib.Nifti1Image(nifti_res_SAMonly_lesionrank, affine=nib.load(nifti_path).affine)
nib.save(nifti_mask, 'output_video_02_1070_2_10152014_23u489123u+dir={0}_rot90_sym_rawdata_37489237_1Samfilt-2_lesionrank_skullstripped.nii.gz'.format(n))

video_dir = output_dir
# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


'''





# Implement later
##bias_corrected = nib.Nifti1Image(image_data, affine=input_image.affine)
##output_path = "segmentation_mask_volume_01_1007_20140402_SAM2_MPRAGE_bias_corrected.nii.gz"  # Replace with your desired output path
##nib.save(bias_corrected, output_path)

# Normalize the image data to 0-255 for SAM2 compatibility
image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255
image_data = image_data.astype(np.uint8)

mask_volume = np.zeros(image_data.shape, dtype=np.uint8)

predictor = SAM2ImagePredictor(sam2_model)

#for i in range(image_data.shape[-1]):  # Iterate through slices along the z-axis
# Select a slice to visualize and segment
start = 0#image_data.shape[2] // 2  # Middle slice as an example

# these masks represent each of 3 directions - when dot-multiplied with the index
# they allow for segmentation in 3 dimensions
# 0: axial
# 1: coronal
# 2: sagittal
for n in range(0,3):
    if n is 0:
        start = 128
    if n is 1:
        start = 128
    if n is 2:
        start = 128
    for slice_idx in range(start,start+5):#image_data.shape[n]):#image_data.shape[2]):
        print('{0} / {1}'.format(slice_idx,image_data.shape[n]))
        if n is 0:
            slice_2d = image_data[slice_idx,:,:]
        if n is 1:
            slice_2d = image_data[:,slice_idx,:]
        if n is 2:
            slice_2d = image_data[:,:,slice_idx]

        # Cap the maximum brightness for segmentation
        #slice_2d[np.where(slice_2d > 80)] = 80
        # Set dim regions to 0
        #slice_2d[np.where(slice_2d < 85)] = 0
        # Convert the slice to RGB format for SAM
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)
        predictor.set_image(slice_rgb)

        tolerance = 170 # threshold under which re-segmentation ceases
        iterations = 1 # the maximum number of segmentation attempts

        max_val = 1e100
        iteration = 0
        curr_slice_2d = slice_2d
        curr_mask_0 = np.zeros(np.shape(slice_2d))
        while max_val > tolerance and iteration < iterations:
            # Define a point (or points) to guide the segmentation
            max_val = np.max(curr_slice_2d)
            print('max: {0}'.format(max_val))

            # Gaussian blur the image (equivalent of taking the avg of 5x5 region)
            sigma = 3  # Standard deviation for Gaussian kernel
            blurred_image = gaussian_filter(curr_slice_2d, sigma=sigma)

            # Find the brightest region
            brightest_index = np.unravel_index(np.argmax(blurred_image), blurred_image.shape)
            p =np.arr(np.unravel_index(np.argmax(curr_slice_2d), slice_2d.shape));
            #p=brightest_index
            input_point = np.array([[p[1],p[0]]])  # Example: one point at (100, 150)
            input_label = np.array([1])  # 1 indicates foreground
            print(input_point)

            # Perform segmentation
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True  # Return multiple masks for more options
            )

            # Convert the slice to RGB format for SAM
            slice_rgb = np.stack([curr_slice_2d] * 3, axis=-1)
            predictor.set_image(slice_rgb)

            # Update masks
            curr_slice_2d = (np.multiply(curr_slice_2d,np.invert(masks[0].astype(bool)))).astype(np.uint8)
            curr_mask_0 = np.array(curr_mask_0.astype(bool) | masks[0].astype(bool)).astype(np.uint8)
            # Update tracker
            iteration = iteration + 1
            print('iteration: {0} | tolerance: {1} / {2}'.format(iteration,max_val,tolerance))

        # Now take that output and determine symmetry of current slice to improve prediction quality
        curr_mask_0_symmetrycheck = curr_mask_0#np.multiply(curr_mask_0.astype(np.uint8),compute_asymmetry_index(slice_2d).astype(np.uint8)).astype(np.uint8)

        #Display the results
        fig, ax = plt.subplots(1, len(masks) + 1, figsize=(20, 10))
        ax[0].imshow(slice_2d, cmap="gray")
        ax[0].set_title("Original Slice")
        ax[0].axis("off")
        i=0
        ax[i + 1].imshow(slice_2d, cmap="gray")
        ax[i + 1].imshow(curr_mask_0_symmetrycheck, alpha=0.5, cmap="jet")
        ax[i + 1].set_title(f"Mask {i + 1} (Score: {scores[0]:.2f})")
        ax[i + 1].axis("off")

        plt.show()

        # Save the masks as binary images (optional)
        saveBinary = False;
        if saveBinary:
            for i, mask in enumerate(masks):
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(f"segmentation_mask_{i + 1}_slice_{slice_idx}.png")

        # Save the masks as a NIfTI file (optional)
        if n is 0:
            mask_volume[slice_idx,:,:] = curr_mask_0_symmetrycheck.astype(np.uint8)
        if n is 1:
            mask_volume[:,slice_idx,:] = curr_mask_0_symmetrycheck.astype(np.uint8)
        if n is 2:
            mask_volume[:,:,slice_idx] = curr_mask_0_symmetrycheck.astype(np.uint8)

    # Save the resulting mask volume as a NIfTI file
    mask_nifti = nib.Nifti1Image(mask_volume, affine=nib.load(nifti_path).affine)
    output_path = "segmentation_mask_volume_01_1007_20140402_SAM2_MPRAGE_dir={0}_1.nii.gz".format(n)  # Replace with your desired output path
    ####nib.save(mask_nifti, output_path)

    print(f"Saved the segmented mask volume to {output_path}")
'''
