clear; close all;

% Task 5: Robust method --------------------------

% Load the new image
IMG = imread('IMG_14.png');

% Convert image to grayscale
I_gray = rgb2gray(IMG);

% Rescale image
height = 512;
[rows, cols] = size(I_gray);
new_width = round(cols * (height / rows));
I_resized = imresize(I_gray, [height, new_width]);

% Adaptive Histogram Equalization
I_enhanced = adapthisteq(I_gray, 'NumTiles', [32 32], 'ClipLimit', 0.012);

% Apply Gaussian smoothing
I_smoothed = imgaussfilt(I_enhanced, 2);  % Gaussian filter with sigma=2

gamma = 0.8;  % Gamma < 1 brightens, Gamma > 1 darkens
I_gamma = imadjust(I_smoothed, [], [], gamma);

% Normalize to range [0, 1]
I_normalized = mat2gray(I_gamma);

% Otsu's Method
thresh = graythresh(I_normalized); 
I_binary = imbinarize(I_normalized, thresh);

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
mask(25:end-25, 25:end-25) = 1;  % Initial mask (avoid borders)
bw = activecontour(I_binary, mask, 500); % Segment with active contour
segmented = bwareaopen(bw, 700); % Remove small objects (<700 pixels)
figure, imshow(segmented);
title('Active Contour Segmentation');

% Filling Image

%Add Padding
padded_edges = padarray(segmented, [1, 1], 0, 'both');
se = strel('disk', 3);

%Dilate
a_dilated_edges = imdilate(padded_edges, se);

%Bridge
dilated_edges = bwmorph(a_dilated_edges,"bridge");


%a: Filling the cells at top and left border of the image
d_edges_a = padarray(dilated_edges,[1 1],1,'pre');
d_edges_a_filled = imfill(d_edges_a, "holes");
d_edges_a_filled = d_edges_a_filled(2:end, 2:end);

%b: Filling the cells at top and right border of the image
d_edges_b = padarray(padarray(dilated_edges,[1 0],1,'pre'),[0 1],1,'post');
d_edges_b_filled = imfill(d_edges_b, "holes");
d_edges_b_filled = d_edges_b_filled(2:end, 1:end-1);

%c: Filling the cells at bottom and right border of the image
d_edges_c = padarray(dilated_edges,[1 1],1,'post');
d_edges_c_filled = imfill(d_edges_c, "holes");
d_edges_c_filled = d_edges_c_filled(1:end-1,1:end-1);

%d: Filling the cells at top and left border of the image
d_edges_d = padarray(padarray(dilated_edges,[1 0],1,'post'),[0 1],1,'pre');
d_edges_d_filled = imfill(d_edges_d, "holes");
d_edges_d_filled = d_edges_d_filled(1:end-1, 2:end);

%Fill image
I_filled = d_edges_a_filled | d_edges_b_filled | d_edges_c_filled | d_edges_d_filled;

% Remove very small objects
I_cleaned = bwareaopen(I_filled, 1000);

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
blood_cell_mask = bwareaopen(imopen(logical(I_cleaned), strel('disk', 14)), 3000);
bacteria_mask = I_cleaned & ~blood_cell_mask;
bacteria_mask = bwareaopen(bacteria_mask, 300); 

% Combine labeled masks into an RGB image
colored_final = zeros([size(I_cleaned), 3]);
colored_final(:,:,1) = blood_cell_mask; % Red for blood cells
colored_final(:,:,2) = bacteria_mask;   % Green for bacteria
colored_final(:,:,3) = bacteria_mask;   % Blue for bacteria
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