##########################################################################
# Mean Shift Algorithm for Image Filtering (Smoothing) and Segmentations
# Optimized Version - Faster Performance
# By Masoud Rafiee & Sonia Tayeb Cherif
# Final Project of Fall 2025 CS463/CS563 (Computer Vision)
##########################################################################

import os  # os path file (handle file paths and image files existence)
import numpy as np  # numerical arrays and math
from PIL import Image  # load/write PPM/PGM files & save results
import \
    matplotlib.pyplot as plt  # to display originial & proccessed images in windows on screen (render arrays as visuals)
import sys  # to handle program exits cleanly& manage user i/o ops safely
from numba import jit, prange  # JIT compilation for massive speedup
from scipy import ndimage  # optimized C implementations for region operations


# separating logic from execution
def main():
    print("\n\n=== MEAN SHIFT IMAGE PROCESSING (OPTIMIZED) ===")
    print("This program performs image filtering (smoothing) and segmentation using Mean Shift Algorithm.\n")
    image_name = input("Enter the Image File Name (e.g. image.ppm): ")
    image_path = os.path.join("final_project_images", image_name)
    if not os.path.exists(image_path):
        print(f"Error! -> the image file '{image_path}' not found!")
        sys.exit(1)  # terminate the program immediately with error code 1 (prevent crash)
    img = Image.open(image_path)  # load the image into a PIL image obj (open and load and ready to process)
    img_array = np.array(img)  # convert PIL image into NumPy array of pixel values
    print(f"Image Loaded Successfully: {img_array.shape}")  # display image dimensions (H/W/Channels)
    print("\nSelect Operation:")
    print("1. Filtering (Smoothing)")
    print("2. Segmentation")
    choice = input("Enter your choice (1 or 2): ")
    if choice == "1":  # starting for filtering-sepcific params (hs,hr)
        hs = float(input(
            "Enter spatial bandwidth (hs) 7-20 pixels: "))  # radius in pixel space - convert str to float (for distance calculation)
        hr = float(input(
            "Enter range bandwidth (hr) 15-40 (for 0-255 scale): "))  # radius in color/intensity space - convert to float for measuring color similarity between pixels
        M = None  # since filtering doesnt use region elimination

    # branch to segmentation param collection if user chose 2
    elif choice == "2":
        hs = float(input("Enter Spatial Bandwidth (hs) 7-20 pixels: "))  # affects region size
        hr = float(input("Enter Range Bandwidth (hr) 15-40 (for 0-255 scale): "))  # affects color grouping
        M = int(input("Enter Minimum Region Size (M) 50-200 pixels: "))  # to eliminate noise regions
    else:
        print("Invalid Choice!! Please Enter 1 or 2.")
        sys.exit(1)  # exit cleanly rather than continue with undefined params

    # for feautre space selection we need to see if the image is 3D (color) or 2D (grayscale) arrays
    is_color = len(img_array.shape) == 3
    print(
        f"\nProcessing {'color' if is_color else 'grayscale'} image...")  # inform usr which type detected and started processing

    if M is None:
        # means we should call filtering function and store smoothed result
        result = mean_shift_filter(img_array, hs, hr, is_color)
        title = "Filtered (Smoothed) Image"  # setting display title for filtering
    else:
        # call segmentation function with region elimination
        result = mean_shift_segment(img_array, hs, hr, M, is_color)
        title = "SEGMENTED Image"  # setting display title for segmentation

    # display original and result side by side for comparing
    display_results(img_array, result, title)

    # create RESULTS folder if it doesnt exist
    results_folder = "RESULTS"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)  # create the folder

    # save the result image to RESULTS folder
    output_filename = f"output_{choice}_{'filter' if M is None else 'segment'}_{image_name}"
    output_path = os.path.join(results_folder, output_filename)  # path to RESULTS folder
    result_img = Image.fromarray(result.astype(np.uint8))  # convert numpy array back to PIL Image
    result_img.save(output_path)  # save the image file
    print(f"Result saved as: {output_path}")  # confirm save location
    print("Processing Complete !!!")


########### THE FUNCTIONS (OPTIMIZED) ###########

# designing the Mean Shift Algorithm that smooths image while preserving edges (now faster with pre-computed kernals)
def mean_shift_filter(img, hs, hr, is_color):
    # extracting the height and width from image shape (works both for 2d and 3d)
    height, width = img.shape[:2]
    result = np.copy(img).astype(np.float64)  # independent copy as float64 for precise decimal calc during averaging
    print("Applying Mean Shift filtering to each pixel...")

    # Pre-compute spatial kernel once (OPTIMIZATION: dont recalculate distances for every pixel every iteration!)
    kernel_size = int(2 * hs + 1)  # total size of kernal window
    y_grid, x_grid = np.ogrid[-hs:hs + 1, -hs:hs + 1]  # create coordinate grids centered at origin
    spatial_distances = np.sqrt(y_grid ** 2 + x_grid ** 2)  # eucildean distances from center for entire kernal
    spatial_mask = spatial_distances <= hs  # boolean mask for circular window (only pixels within radius)
    spatial_weights = np.exp(
        -(spatial_distances ** 2) / (2 * hs ** 2)) * spatial_mask  # pre-calcualte gaussian weights once and apply mask

    # Pre-compute constants
    hr_sq_2 = 2 * hr ** 2

    if is_color:
        result = process_color_parallel(img, result, hs, hr, hr_sq_2, spatial_weights, spatial_mask, height, width)
    else:
        result = process_gray_parallel(img, result, hs, hr, hr_sq_2, spatial_weights, spatial_mask, height, width)

    return result.astype(np.uint8)  # converting back to unit8 (0-255) for valid image format and return


# Ultra-fast parallel processing for color images
@jit(nopython=True, parallel=True, cache=True)
def process_color_parallel(img, result, hs, hr, hr_sq_2, spatial_weights, spatial_mask, height, width):
    for i in prange(height):  # loop through each row to proccess each pixel
        if i % 50 == 0:  # print progress every 50 rows to show work without flooding output
            print(f" Processing row {i}/{height}")
        for j in range(width):  # for each col, combined with row loop (covering all pixels)
            result[i, j] = mean_shift_pixel_color(img, i, j, hs, hr, hr_sq_2, spatial_weights, spatial_mask, height,
                                                  width)  # pass pre-computed weights to avoid recalc
    return result


# Ultra-fast parallel processing for grayscale images
@jit(nopython=True, parallel=True, cache=True)
def process_gray_parallel(img, result, hs, hr, hr_sq_2, spatial_weights, spatial_mask, height, width):
    for i in prange(height):  # loop through each row to proccess each pixel
        if i % 50 == 0:  # print progress every 50 rows to show work without flooding output
            print(f" Processing row {i}/{height}")
        for j in range(width):  # for each col, combined with row loop (covering all pixels)
            result[i, j] = mean_shift_pixel_gray(img, i, j, hs, hr, hr_sq_2, spatial_weights, spatial_mask, height,
                                                 width)  # pass pre-computed weights to avoid recalc
    return result


# Mean Shift Convergance: for single pixel by iteratively shifting toward density peak (OPTIMIZED with vectorization!)
@jit(nopython=True, cache=True)
def mean_shift_pixel_color(img, y, x, hs, hr, hr_sq_2, spatial_weights, spatial_mask, height, width):
    # Calculate window boundaries (same as before)
    y_min = max(0, int(y - hs))  # calcualte vertical search window bounds, clamped to image boundaries
    y_max = min(height, int(y + hs + 1))
    x_min = max(0, int(x - hs))  # calcualte horzonital search window bounds (claped to image bonudaries again)
    x_max = min(width, int(x + hs + 1))

    # Extract window (get all neightbor pixels at once instead of looping!)
    window = img[y_min:y_max, x_min:x_max].astype(
        np.float64)  # extract entire window as array for vectorized operations

    # Adjust spatial kernel for boundary cases (edge pixels have smaller windows)
    kernel_y_start = int(hs) - (y - y_min)  # calculate offset into pre-computed kernal
    kernel_y_end = kernel_y_start + (y_max - y_min)  # end position in kernal
    kernel_x_start = int(hs) - (x - x_min)  # horzonital offset
    kernel_x_end = kernel_x_start + (x_max - x_min)  # horzonital end

    current_spatial_weights = spatial_weights[kernel_y_start:kernel_y_end,
                              kernel_x_start:kernel_x_end]  # slice pre-computed weights to match window size
    current_spatial_mask = spatial_mask[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]  # slice mask too

    current = img[y, x].astype(np.float64)  # get starting pixel and turn it into gfloat64 for same reason

    # FOR SPEED CHANGE HERE :
    ###################################
    threshold = 2.0  # convergence threshold increased for faster but still good results
    max_iterations = 10  # limit iterations to prevent taking forever on difficult pixels
    ##################

    tiny_threshold = 0.3
    consecutive_tiny = 0

    for iteration in range(max_iterations):  # exit when converged or hit max
        # Vectorized range distance computation (calculate all distances at once! MUCH FASTER)
        range_distances = np.sqrt(
            np.sum((window - current) ** 2, axis=2))  # eucildean dist in color space for all pixels simultaniously

        # Create range mask (which neightbors pass color similarity test)
        range_mask = range_distances <= hr  # boolean array marking pixels within range bandwidth
        combined_mask = current_spatial_mask & range_mask  # combine spatial and range masks (must pass both tests)

        # If no neighbors qualify, return current value (edge case handling)
        if not np.any(combined_mask):
            return current  # no valid neightbors so stop here

        # Vectorized weight computation (calculate all weights at once instead of looping!)
        range_weights = np.exp(-(range_distances ** 2) / hr_sq_2)  # gaussian weights based on color similarity
        weights = current_spatial_weights * range_weights * combined_mask  # multiply spatial, range weights and apply mask

        # Compute weighted mean (vectorized summation much faster than loops)
        sum_weights = np.sum(weights)  # total weight for normalization

        if sum_weights > 0:
            # For color images, weight each channel (RGB handled simultaniously)
            new_value = np.zeros(3, dtype=np.float64)
            for c in range(3):
                new_value[c] = np.sum(window[:, :, c] * weights) / sum_weights  # broadcast weights to 3 channels
        else:
            new_value = current  # fallback if no weights (shouldnt happen due to check above)

        # Check convergence (same as before)
        shift = np.sqrt(np.sum((new_value - current) ** 2))  # calculate how much pixel value shifted in this iteration

        if shift < tiny_threshold:
            consecutive_tiny += 1
            if consecutive_tiny >= 2:
                break
        else:
            consecutive_tiny = 0

        if shift < threshold:  # check if shift smal enough to consider converged
            break  # stop iterating, we found the mode

        current = new_value  # update current new value for next iteration, continue shifting

    return current  # return final converged value


# Grayscale version - even faster
@jit(nopython=True, cache=True)
def mean_shift_pixel_gray(img, y, x, hs, hr, hr_sq_2, spatial_weights, spatial_mask, height, width):
    # Calculate window boundaries (same as before)
    y_min = max(0, int(y - hs))  # calcualte vertical search window bounds, clamped to image boundaries
    y_max = min(height, int(y + hs + 1))
    x_min = max(0, int(x - hs))  # calcualte horzonital search window bounds (claped to image bonudaries again)
    x_max = min(width, int(x + hs + 1))

    # Extract window (get all neightbor pixels at once instead of looping!)
    window = img[y_min:y_max, x_min:x_max].astype(
        np.float64)  # extract entire window as array for vectorized operations

    # Adjust spatial kernel for boundary cases (edge pixels have smaller windows)
    kernel_y_start = int(hs) - (y - y_min)  # calculate offset into pre-computed kernal
    kernel_y_end = kernel_y_start + (y_max - y_min)  # end position in kernal
    kernel_x_start = int(hs) - (x - x_min)  # horzonital offset
    kernel_x_end = kernel_x_start + (x_max - x_min)  # horzonital end

    current_spatial_weights = spatial_weights[kernel_y_start:kernel_y_end,
                              kernel_x_start:kernel_x_end]  # slice pre-computed weights to match window size
    current_spatial_mask = spatial_mask[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]  # slice mask too

    current = float(img[y, x])  # get starting pixel and turn it into gfloat64 for same reason

    # FOR SPEED CHANGE HERE :
    ###################################
    threshold = 2.0  # convergence threshold increased for faster but still good results
    max_iterations = 10  # limit iterations to prevent taking forever on difficult pixels
    ##################

    tiny_threshold = 0.3
    consecutive_tiny = 0

    for iteration in range(max_iterations):  # exit when converged or hit max
        # Vectorized range distance computation (calculate all distances at once! MUCH FASTER)
        range_distances = np.abs(window - current)  # absolute diff for grayscale (simpler but same idea)

        # Create range mask (which neightbors pass color similarity test)
        range_mask = range_distances <= hr  # boolean array marking pixels within range bandwidth
        combined_mask = current_spatial_mask & range_mask  # combine spatial and range masks (must pass both tests)

        # If no neighbors qualify, return current value (edge case handling)
        if not np.any(combined_mask):
            return current  # no valid neightbors so stop here

        # Vectorized weight computation (calculate all weights at once instead of looping!)
        range_weights = np.exp(-(range_distances ** 2) / hr_sq_2)  # gaussian weights based on color similarity
        weights = current_spatial_weights * range_weights * combined_mask  # multiply spatial, range weights and apply mask

        # Compute weighted mean (vectorized summation much faster than loops)
        sum_weights = np.sum(weights)  # total weight for normalization

        if sum_weights > 0:
            # For grayscale (simpler calc)
            new_value = np.sum(window * weights) / sum_weights  # weighted average of all pixels
        else:
            new_value = current  # fallback if no weights (shouldnt happen due to check above)

        # Check convergence (same as before)
        shift = abs(new_value - current)  # calculate how much pixel value shifted in this iteration

        if shift < tiny_threshold:
            consecutive_tiny += 1
            if consecutive_tiny >= 2:
                break
        else:
            consecutive_tiny = 0

        if shift < threshold:  # check if shift smal enough to consider converged
            break  # stop iterating, we found the mode

        current = new_value  # update current new value for next iteration, continue shifting

    return current  # return final converged value


# Mean Shift Segmentation by filtering, clustering pixels into regions & eliminating small ones
def mean_shift_segment(img, hs, hr, M, is_color):
    print("Step 1: Filtering image...")
    filtered = mean_shift_filter(img, hs, hr,
                                 is_color)  # apply mean shift filtering first to smooth image and reduce noise
    print(
        "Step 2: Clustering pixels into regions...")  # announce region labeling phase where similar neighboring pixels get grouped

    # Ultra-fast region labeling
    labels = fast_region_label(filtered, hr, is_color)

    num_regions = labels.max()
    print(f"Found {num_regions} initial regions")  # report total regions found before elimination
    print(f"Step 3: Eliminating regions smaller than {M} pixels...")  # announce region filtering phase
    labels = eliminate_small_regions_fast(labels, filtered, M, hr,
                                          is_color)  # merge small noise regions with most similar neightbors
    result = np.zeros_like(filtered)  # create empty array matching filtered dimensions to store final colored segments
    unique_labels = np.unique(labels[labels > 0])  # find all unique region IDs (excluding 0=unlabeled)
    for label in unique_labels:  # loop through each region to calcualte and assign average color
        mask = labels == label  # create boolean mask selecting pixels belonging to current region
        region_mean = filtered[mask].mean(axis=0)  # calculate mean color and intensity of all pixels in this region
        result[mask] = region_mean  # assign uniform color to all pixels in region for visual segmentation
    return result.astype(np.uint8)  # convert to unit8 and return final segmented image with colored regions


# Ultra-fast region labeling
def fast_region_label(img, hr, is_color):
    height, width = img.shape[:2]
    label_img = np.zeros((height, width), dtype=np.int32)
    current_label = 1

    if is_color:
        # Use optimized flood fill for color
        for i in range(height):
            for j in range(width):
                if label_img[i, j] == 0:
                    region_grow_color(img, label_img, i, j, current_label, hr)
                    current_label += 1
    else:
        # For grayscale
        for i in range(height):
            for j in range(width):
                if label_img[i, j] == 0:
                    region_grow_gray(img, label_img, i, j, current_label, hr)
                    current_label += 1

    return label_img


# flood-fill to label connected similar pixels starting from seed position
@jit(nopython=True, cache=True)
def region_grow_color(img, labels, start_y, start_x, label, hr):
    stack = [(start_y, start_x)]  # stack with starting position for iterative flood-fill (avoids recursion overflow)
    seed_value = img[start_y, start_x].astype(np.float64)  # store seed pixel color as reference for similarity compare
    height, width = labels.shape
    hr_sq = hr * hr

    while len(stack) > 0:  # proccess stack until empty (all reachable similar neightbors proccessed)
        y, x = stack.pop()  # pop next pixel coordinates from stack (LIFO order)
        if labels[y, x] != 0:  # skip pixels already labeled to prevent relabeling and infinite loops
            continue
        current_value = img[y, x].astype(np.float64)  # get current pixels color for similarity test with seed

        # Color distance calculation
        distance_sq = 0.0
        for c in range(3):
            diff = current_value[c] - seed_value[c]
            distance_sq += diff * diff

        if distance_sq > hr_sq:  # skip pixels two different from seed (exceeds range bandwidth)
            continue
        labels[y, x] = label  # assign region ID to current pixel since it passed similarity test

        # 4-connected neightbors (up,down,left,right) for expansion
        if y > 0 and labels[y - 1, x] == 0:
            stack.append((y - 1, x))
        if y < height - 1 and labels[y + 1, x] == 0:
            stack.append((y + 1, x))
        if x > 0 and labels[y, x - 1] == 0:
            stack.append((y, x - 1))
        if x < width - 1 and labels[y, x + 1] == 0:
            stack.append((y, x + 1))


# flood-fill for grayscale images
@jit(nopython=True, cache=True)
def region_grow_gray(img, labels, start_y, start_x, label, hr):
    stack = [(start_y, start_x)]  # stack with starting position for iterative flood-fill (avoids recursion overflow)
    seed_value = float(img[start_y, start_x])  # store seed pixel color as reference for similarity compare
    height, width = labels.shape
    hr_sq = hr * hr

    while len(stack) > 0:  # proccess stack until empty (all reachable similar neightbors proccessed)
        y, x = stack.pop()  # pop next pixel coordinates from stack (LIFO order)
        if labels[y, x] != 0:  # skip pixels already labeled to prevent relabeling and infinite loops
            continue
        current_value = float(img[y, x])  # get current pixels color for similarity test with seed

        # Grayscale distance calculation
        diff = current_value - seed_value
        distance_sq = diff * diff

        if distance_sq > hr_sq:  # skip pixels two different from seed (exceeds range bandwidth)
            continue
        labels[y, x] = label  # assign region ID to current pixel since it passed similarity test

        # 4-connected neightbors (up,down,left,right) for expansion
        if y > 0 and labels[y - 1, x] == 0:
            stack.append((y - 1, x))
        if y < height - 1 and labels[y + 1, x] == 0:
            stack.append((y + 1, x))
        if x > 0 and labels[y, x - 1] == 0:
            stack.append((y, x - 1))
        if x < width - 1 and labels[y, x + 1] == 0:
            stack.append((y, x + 1))


# finds regions smaller than M pixels and merges them with most similar neightbors
def eliminate_small_regions_fast(labels, img, M, hr, is_color):
    unique, counts = np.unique(labels[labels > 0],
                               return_counts=True)  # count pixels per region, return_counts gives sizes alongside labels
    small_regions = unique[counts < M]  # filter to keep only regions with fewer than M pixels (noise regions)
    print(f" Eliminating {len(small_regions)} small regions")  # report how many regions will be eliminated

    # Pre-compute region colors for speed
    region_colors = {}
    for region_label in unique:
        mask = labels == region_label
        region_colors[region_label] = img[mask].mean(axis=0)

    for region_label in small_regions:  # loop through each small region to find neightbor and merge
        mask = labels == region_label  # boolean mask selecting pixels in current small region
        region_color = region_colors[region_label]  # average color of small region for similarity compare
        neighbor_labels = set()  # empty set to collect unique neighboring region IDs

        # Fast neighbor finding using dilation
        dilated = ndimage.binary_dilation(mask)
        boundary = dilated & ~mask
        neighbor_labels_array = labels[boundary]
        neighbor_labels = set(neighbor_labels_array[neighbor_labels_array != region_label])
        neighbor_labels.discard(0)

        if not neighbor_labels:  # handle edge case where no neightbors found (isolated region)
            continue
        best_neighbor = None  # variable to track most similar neighboring region
        min_distance = float('inf')  # minimum distance to infinity so first neightbor becomes best by default
        for neighbor_label in neighbor_labels:  # loop through each neighboring region to find closest color match
            neighbor_color = region_colors[neighbor_label]  # neightbor regions average color
            distance = np.linalg.norm(region_color - neighbor_color)  # eucildean distance between region colors
            if distance < min_distance:  # update best match if current neightbor is closer
                min_distance = distance  # track new minimum distance
                best_neighbor = neighbor_label  # store closest neightbors ID
        labels[mask] = best_neighbor  # merge small region into best neightbor by assigning its label
    return labels  # return cleaned label array with small regions merged


# shows originial and proccessed images side-by-side using matplotlib for comparing
def display_results(original, result, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # figure with 1 row, 2 cols for side-by-side display
    axes[0].imshow(original, cmap='gray' if len(
        original.shape) == 2 else None)  # display originial image in first subplot, use gray colormap if grayscale
    axes[0].set_title("Original Image")  # label first subplot as originial
    axes[0].axis('off')  # hide axis numbers and ticks for clean display
    axes[1].imshow(result.astype(np.uint8), cmap='gray' if len(
        result.shape) == 2 else None)  # display result image in second subplot with appropriate colormap
    axes[1].set_title(title)  # label with operation-sepcific title
    axes[1].axis('off')  # hide axis elements for clean display
    plt.tight_layout()  # automatically adjust spacing to prevent overlapping titles
    plt.show()  # open matplotlib window to display figure, blocks until closed


# check if script is run directly (not imported as module)
if __name__ == "__main__":
    main()  # call main function to start program execution
