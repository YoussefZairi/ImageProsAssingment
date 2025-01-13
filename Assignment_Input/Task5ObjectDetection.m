clear; close all;

% Task 5: Robust method --------------------------

% Load the new image
IMG = imread('IMG_06.png');

% Convert image to grayscale
I_gray = rgb2gray(IMG);

% Rescale image
height = 512;
[rows, cols] = size(I_gray);
new_width = round(cols * (height / rows));
I_resized = imresize(I_gray, [height, new_width]);

% Adaptive Histogram Equalization
I_enhanced = adapthisteq(I_gray, 'NumTiles', [32 32], 'ClipLimit', 0.012);

I_smoothed = imgaussfilt(I_enhanced, 2);  % Gaussian filter with sigma=2

gamma = 0.8;  % Gamma < 1 brightens, Gamma > 1 darkens
I_gamma = imadjust(I_smoothed, [], [], gamma);

I_normalized = mat2gray(I_gamma);  % Normalize to range [0, 1]

% Otsu's Method
thresh = graythresh(I_normalized); 
I_binary = imbinarize(I_normalized, thresh);

% Edge Detection
I_edge = edge(I_binary, 'roberts');

% Morphological Processing
se_close = strel('disk',3);
I_closed = imclose(I_edge, se_close); %close small gaps

I_bridged1 = bwmorph(I_closed, "bridge");

%Add Padding
padded_edges = padarray(I_bridged1, [1, 1], 0, 'both');
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

% Remove very small objects with area < 400
I_cleaned = bwareaopen(I_filled, 3000);

% Dilate to separate overlapping cells better
se_dilate = strel('disk', 3);
I_dilated = imerode(I_cleaned, se_dilate);

% Take objects larger than 300 pixels in area.
I_filtered = bwareafilt(I_dilated, [2000, inf]);

figure;
imshow (I_dilated);
title ('Filtered');

% Label connected components
I_labeled = bwlabel(I_filtered);

% Process and filter objects not part of b1
blood_cell_mask = bwareaopen(imopen(logical(I_filtered), strel('disk', 14)), 3000); 
bacteria_mask = I_filtered & ~blood_cell_mask;
bacteria_mask = bwareaopen(bacteria_mask, 300);

L1 = zeros(size(I_filtered)); 
L1(I_filtered) = 1; % Label objects in labeled image as 1

L2 = zeros(size(blood_cell_mask)); 
L2(blood_cell_mask) = 2; % Label objects in blood cell mask as 2

% Combine labeled masks
colored_final = L1 + L2;

% Create an RGB image
colored_final = zeros([size(colored_final), 3]);
colored_final(:,:,1) = blood_cell_mask;
colored_final(:,:,2) = bacteria_mask;
colored_final(:,:,3) = bacteria_mask;
figure, imshow(colored_final);
title('Colored Final')

% Display Results
figure;
subplot(2, 2, 1);
imshow(I_enhanced);
title('Enhanced Image');

subplot(2, 2, 2);
imshow(I_normalized);
title('Normalized Image');

subplot(2, 2, 3);
imshow(I_binary);
title('Binarized Image');

subplot(2, 2, 4);
imshow(I_edge);
title('Edge Detection');

% Display Results 2
figure;
subplot(2, 2, 1);
imshow(I_closed);
title('Closed');

subplot(2, 2, 2);
imshow(I_cleaned);
title('Cleaned');

subplot(2, 2, 3);
imshow(I_filled);
title('Filled');

subplot(2, 2, 4);
imshow(colored_final);
title('Colored Image');

% Task 6: Performance evaluation -----------------
%Step 1: Load ground truth data
GT = imread("IMG_06_GT.png");
% To visualise the ground truth image, you can
% use the following code.
L_GT = label2rgb(rgb2gray(GT), 'prism','k','shuffle');
figure, imshow(L_GT)