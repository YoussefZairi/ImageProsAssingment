clear; close all;

% Task 5: Robust method --------------------------

% Load the new image
IMG = imread('IMG_01.png');

% Convert image to grayscale
I_gray = rgb2gray(IMG);

% Rescale image
height = 512;
[rows, cols] = size(I_gray);
new_width = round(cols * (height / rows));
I_resized = imresize(I_gray, [height, new_width]);

% Adaptive Histogram Equalization
I_enhanced = adapthisteq(I_gray, 'NumTiles', [64 64], 'ClipLimit', 0.01);

% Apply Gaussian smoothing
I_smoothed = imgaussfilt(I_enhanced, 2);  % Gaussian filter with sigma=2

gamma = 0.8;  % Gamma < 1 brightens, Gamma > 1 darkens
I_gamma = imadjust(I_smoothed, [], [], gamma);

% Normalize to range [0, 1]
I_normalized = mat2gray(I_gamma);

% Otsu's Method
thresh = graythresh(I_normalized); 
I_binary = imbinarize(I_normalized, 'adaptive', 'sensitivity', 0.2);

% Step 1: Edge Detection
I_edge = edge(I_binary, 'roberts'); % Detect edges
figure, imshow(I_edge);
title('Edge Detection');

% Step 2: Morphological Processing
se_close = strel('disk', 3);
I_closed = imclose(I_edge, se_close); % Close small gaps
I_bridged1 = bwmorph(I_closed, "bridge"); % Bridge gaps in edges
figure, imshow(I_bridged1);
title('Bridged Edges');

% Step 3: Segmentation using Active Contour
mask = zeros(size(I_binary)); 
mask(50:end-50, 50:end-50) = 1;  % Initial mask (avoid borders)
bw = activecontour(I_binary, mask, 1000); % Segment with active contour
segmented = bwareaopen(bw, 700); % Remove small objects (<700 pixels)
figure, imshow(segmented);
title('Active Contour Segmentation');

% Filling Image

% Add padding to the top and bottom edges
top_bottom_padding = padarray(segmented, [1, 0], 1, 'both');  % Add rows
top_bottom_filled = imfill(top_bottom_padding, 'holes');      % Fill holes

% Add padding to the left and right edges
left_right_padding = padarray(segmented, [0, 1], 1, 'both');  % Add columns
left_right_filled = imfill(left_right_padding, 'holes');      % Fill holes

% Crop both filled images back to the original size
top_bottom_cropped = top_bottom_filled(2:end-1, :);
left_right_cropped = left_right_filled(:, 2:end-1);

% Combine the cropped filled images
combined_filled = top_bottom_cropped | left_right_cropped;

% Dilate the combined edges
se = strel('disk', 2);
dilated_edges = imdilate(combined_filled, se);

% Fill all holes in the combined image
I_filled = imfill(dilated_edges, 'holes');

% Display the final filled image
figure;
imshow(I_filled);
title('Final Selective Edge Padding and Filling');

% Remove very small objects
I_cleaned = bwareaopen(I_filled, 400);

figure;
imshow(I_cleaned);
title('Cleaned');

% Further refine objects to separate overlapping cells
se_dilate = strel('disk', 3);
BW_dilated = imerode(I_cleaned, se_dilate);

% Filter objects by size (2000 pixels and larger)
BW_filtered = bwareafilt(BW_dilated, [2000, inf]);

% Display final segmentation results
figure, imshow(BW_filtered);
title('Filtered Segmentation');

% Label connected components
I_labeled = bwlabel(BW_filtered);

% Step 5: Label Objects
% Separate blood cells and bacteria using size and shape
blood_cell_mask = bwareaopen(imopen(logical(I_cleaned), strel('disk', 14)), 2500);
bacteria_mask = I_cleaned & ~blood_cell_mask;
bacteria_mask = bwareaopen(bacteria_mask, 300);

% Combine labeled masks into an RGB image
colored_final = zeros([size(I_cleaned), 3]);
colored_final(:,:,1) = blood_cell_mask; % Red for blood cells
colored_final(:,:,2) = bacteria_mask;   % Green for bacteria
colored_final(:,:,3) = bacteria_mask;   % Blue to combine w green and cyan for bacteria
figure, imshow(colored_final);
title('Colored Final');

% Display Results
figure;
subplot(2, 2, 1);
imshow(I_enhanced);
title('Enhanced Image');

subplot(2, 2, 2);
imshow(I_normalized);
title('Normalized Image');

subplot(2, 2, 3);
imshow(segmented); % Segmented image
title('Segmented Image');

subplot(2, 2, 4);
imshow(BW_filtered);
title('Filtered Segmentation');

% Task 6: Performance evaluation -----------------
% Step 1: Load ground truth data
GT = imread("IMG_14_GT.png");
L_GT = label2rgb(rgb2gray(GT), 'prism', 'k', 'shuffle');

subplot(2,1,1);
imshow(L_GT);
title('Ground Truth')
subplot(2,1,2);
imshow(colored_final);
title('Colored Final');