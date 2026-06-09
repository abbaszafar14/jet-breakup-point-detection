# Jet Breakup Point Detection

Image processing code for detecting breakup points in planar Mie scattering images from jet in crossflow experiments.

## Description

This MATLAB code analyzes Mie scattering images to identify and locate the breakup point in a jet in crossflow configuration.

## Requirements

- MATLAB R2019a or later
- Image Processing Toolbox
- Computer Vision Toolbox

## Usage

1. Prepare your Mie scattering image
2. Run the breakup detection function:
```matlab
   [breakup_point] = breakup_detection(image);
```

## Inputs

- `image`: Grayscale or color image of Mie scattering

## Outputs

- `breakup_point`: Coordinates of detected breakup point

## Files

- `src/breakup_detection.m` - Main detection function
- `examples/sample_usage.m` - Example usage


