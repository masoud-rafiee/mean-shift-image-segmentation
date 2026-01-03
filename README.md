# Mean Shift Image Processing

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

A Python implementation of the **Mean Shift Algorithm** for edge-preserving image filtering and unsupervised image segmentation. This project was developed as the final project for CS563/463 Computer Vision course at Bishop's University (Fall 2025).

## Overview

Mean Shift is a non-parametric clustering technique that iteratively shifts data points towards the densest regions in feature space. This implementation applies Mean Shift to:
- **Image Filtering (Smoothing)**: Reduces noise while preserving sharp edges
- **Image Segmentation**: Groups pixels into coherent regions based on spatial proximity and color similarity

## Features

- ✅ Edge-preserving image smoothing
- ✅ Unsupervised image segmentation with region merging
- ✅ Support for both **grayscale (PGM)** and **color (PPM)** images
- ✅ Optimized with **Numba JIT compilation** for faster performance
- ✅ Interactive parameter tuning (spatial bandwidth, range bandwidth, minimum region size)
- ✅ Automatic result visualization and saving

## Algorithm Details

### Mean Shift Filtering
Smooths images by iteratively shifting each pixel toward the local density maximum in a combined spatial-range feature space:

- **Spatial bandwidth (hs)**: Controls the neighborhood size in the image plane
- **Range bandwidth (hr)**: Controls similarity threshold in intensity/color space

Each pixel is updated using a weighted mean of nearby similar pixels, producing noise reduction without edge blurring.

### Mean Shift Segmentation
Builds on filtering to partition images into regions:

1. **Filtering**: Apply Mean Shift smoothing to reduce noise
2. **Region Growing**: Use flood-fill to cluster similar neighboring pixels
3. **Region Merging**: Eliminate regions smaller than **M** pixels by merging with the most similar neighbor

## Installation

### Prerequisites
- Python 3.8 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/masoud-rafiee/mean-shift-image-segmentation.git
cd mean-shift-image-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Program

```bash
python MeanShift.py
```

### Interactive Workflow

1. **Enter image filename** (e.g., `image1.pgm`)
2. **Select operation**:
   - `1` for Filtering (Smoothing)
   - `2` for Segmentation
3. **Enter parameters**:
   - **hs** (spatial bandwidth): Recommended 7-20 pixels
   - **hr** (range bandwidth): Recommended 15-40 for 0-255 scale
   - **M** (minimum region size, segmentation only): Recommended 50-200 pixels

4. Results are displayed side-by-side with the original image and automatically saved to the `RESULTS/` folder

### Example Commands

**Filtering Example:**
```
Enter the Image File Name: image1.pgm
Select Operation: 1
Enter spatial bandwidth (hs): 10
Enter range bandwidth (hr): 22
```

**Segmentation Example:**
```
Enter the Image File Name: image7.ppm
Select Operation: 2
Enter Spatial Bandwidth (hs): 10
Enter Range Bandwidth (hr): 24
Enter Minimum Region Size (M): 160
```

## Parameter Guidelines

| Parameter | Type | Range | Effect |
|-----------|------|-------|--------|
| **hs** | Spatial bandwidth | 7-20 | Larger values → more spatial smoothing |
| **hr** | Range bandwidth | 15-40 | Larger values → more color merging |
| **M** | Min region size | 50-200 | Larger values → fewer, larger segments |

## Project Structure

```
mean-shift-image-segmentation/
├── MeanShift.py                    # Main implementation
├── README.md                        # Documentation
├── requirements.txt                 # Python dependencies
├── final_project_images/            # Input test images
├── RESULTS/                         # Output processed images
├── Report_MeanShift_$Masoud_Sonia.pdf  # Detailed project report
└── .gitignore
```

## Performance Notes

- The algorithm is computationally intensive due to iterative per-pixel processing
- **Optimizations applied**:
  - Numba JIT compilation for 10-50x speedup
  - Pre-computed spatial kernels
  - Vectorized operations
  - Parallel processing with `prange`
  - Scipy's optimized C implementations for region operations

- Processing time varies with image size (several seconds to minutes)
- Recommended for small to medium resolution images

## Results

### Filtering Results
- Effective noise reduction while preserving edges
- Maintains sharp boundaries (nose contours, facial ridges, eye boundaries)
- Smooths high-frequency texture regions

### Segmentation Results
- Clear partitioning into coherent regions
- Successful elimination of small noisy regions
- Meaningful grouping based on spatial proximity and color similarity

*(See full report for detailed results and analysis)*

## Technical Details

### Feature Spaces
- **Grayscale**: 3D feature space (intensity + x + y coordinates)
- **Color**: 5D feature space (R, G, B + x + y coordinates)

### Libraries Used
- **NumPy**: Numerical array operations
- **Pillow (PIL)**: Image loading and saving
- **Matplotlib**: Result visualization
- **Numba**: JIT compilation for performance
- **SciPy**: Optimized image processing operations

## Authors

- **Masoud Rafiee** - [mrafiee22@ubishops.ca](mailto:mrafiee22@ubishops.ca)
- **Zouina Sonia Tayeb Cherif** - [ztayeb23@ubishops.ca](mailto:ztayeb23@ubishops.ca)

**Course**: CS563/463 - Computer Vision & Image Processing  
**Institution**: Bishop's University  
**Instructor**: Dr. Elmehdi Aitnouri  
**Term**: Fall 2025

## References

- Comaniciu, D., & Meer, P. (2002). Mean shift: A robust approach toward feature space analysis. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
- Course materials from CS563/463 Computer Vision, Bishop's University

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dr. Elmehdi Aitnouri for guidance and course instruction
- Bishop's University Computer Science Department
- Original Mean Shift algorithm developers Comaniciu & Meer

---

**Note**: This implementation prioritizes clarity and educational value while incorporating performance optimizations. For production use, consider additional optimizations like GPU acceleration or hierarchical approaches.