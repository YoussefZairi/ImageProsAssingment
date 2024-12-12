clear; close all;
% Task 5: Robust method --------------------------

% Load the new image
IMG = imread('IMG_02.png');

%Covert image to grayscale
I_gray = rgb2gray(IMG);
figure, imshow(I_gray);
title('Grayscale Image');

%Rescale image
height = 512;
[rows,cols] = size(I_gray);
new_width = round(cols * (height / rows));
I_resized = imresize(I_gray, [height, new_width]);

%Enhance Image
I_enhanced = imadjust(I_resized);

% Calculate mean brightness
mean_brightness = mean(I_enhanced(:)) / 255; % Mean pixel intensity across the whole image normalized [0,1]
disp(['Mean Brightness: ', num2str(mean_brightness)]);
alpha = 0.5;
% Define a threshold to decide binarization method (adjust as needed)
brightness_threshold = 17; % Example threshold (range 0-255)
%deTErmined through finding an image where otsu's method worked correctly
%(i.e img_01 has a mean brightness of Mean Brightness: 17.6348.
%Determine threshold with Otsu's method
threshold = graythresh(I_enhanced);

%Convert to Binary
%%%I_binary = imbinarize(I_enhanced, threshold);

% Select binarization method based on mean brightness
if mean_brightness > brightness_threshold
    % Bright image: Use global thresholding
    disp('Using Global Thresholding...');
    threshold = graythresh(I_gray); % Compute global threshold (Otsu's method)
    I_binary = imbinarize(I_gray, threshold); % Binarize
else
    % Dark image: Use adaptive thresholding
    disp('Using Adaptive Thresholding...');
    sensitivity = alpha * (1-mean_brightness); % Adjust sensitivity (higher for darker images)
    adaptive_thresh = adaptthresh(I_gray, sensitivity);
    I_binary = imbinarize(I_gray, adaptive_thresh); % Binarize
end
%sensitivity = 0.2;
%adaptive_thresh = adaptthresh(I_enhanced, sensitivity, 'ForegroundPolarity','bright');
%I_binary = imbinarize(I_enhanced, adaptive_thresh);
%%%%%%%%%%IMPLEMENT SEPERATION MORPHOLIGICAL METHODS%%%%%%%%%%%
figure; imshow(I_binary);
title('Binarized Image');

I_seperated = bwmorph(I_binary, 'shrink', 4);
figure; imshow(I_seperated);
title('Seperated');

se = strel('disk', 1);  % Define a small disk-shaped structuring element
I_eroded = imerode(I_binary, se);  % Erode the image to create gaps

I_seperated_eroded = bwmorph(I_eroded, 'shrink', 3);

I_edge = edge(I_binary, 'roberts');

%Add Padding
padded_edges = padarray(I_edge, [1, 1], 0, 'both');
se = strel('disk', 3);
%Dilate
a_dilated_edges = imdilate(padded_edges, se);
%Bridge
dilated_edges = bwmorph(a_dilated_edges,"bridge");

%FILL CORNERS
%Fill Corners (ref
%https://blogs.mathworks.com/steve/2013/09/05/defining-and-filling-holes-on-the-border-of-an-image/)

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

figure, imshow(I_filled);
title('Filled')
% Remove noise
I_cleaned = bwareaopen(I_filled, 600);
figure, imshow(I_cleaned);
title('Cleaned')
% Smooth edges
I_smoothed = imclose(I_cleaned, strel('disk', 1));

% Label connected components
I_labeled = bwlabel(I_smoothed);
figure, imshow(I_labeled);
title('Labeled')

%Better Extract Features for robust
% Extract features
stats = regionprops(I_labeled, 'Area', 'Eccentricity', 'Solidity', 'Perimeter');
areas = [stats.Area];
eccentricities = [stats.Eccentricity];
solidities = [stats.Solidity];
perimeters = [stats.Perimeter];
circularities = (4 * pi * areas) ./ (perimeters .^ 2);

%Print stats
for i = 1:numel(stats)
    fprintf('Object %d: Area = %.2f, Eccentricity = %.2f, Solidity = %.2f, Perimeter = %.2f, Circularity: %.2f\n', i, stats(i).Area, stats(i).Eccentricity, stats(i).Solidity, stats(i).Perimeter, circularities(i));
end

% Thresholds for classification
blood_cells_mask = ismember(I_labeled, find([stats.Area] > 2000 & circularities > 0.6 ));
bacteria_mask = ismember(I_labeled, find([stats.Area] > 500 & circularities < 0.6 ));

final_filtered = blood_cells_mask | bacteria_mask;

% Create an RGB image
colored_final = zeros([size(final_filtered), 3]);
colored_final(:,:,1) = blood_cells_mask;
colored_final(:,:,2) = bacteria_mask;
colored_final(:,:,3) = bacteria_mask;
figure, imshow(colored_final);
title('Colored Final')

figure;
subplot(2, 2, 1);
imshow(I_enhanced);
title('Enhanced');

subplot(2, 2, 2);
imshow(I_binary);
title('Binarized');

subplot(2, 2, 3);
imshow(I_edge);
title('Edge Detection');

subplot(2, 2, 4);
imshow(colored_final);
title('Colored Final')

figure; imshow(I_seperated);
title('Seperated');
figure; imshow(I_eroded);
title('Eroded')
figure; imshow(I_seperated_eroded);
title('Seperated - Eroded');

% Task 6: Performance evaluation -----------------
% Step 1: Load ground truth data
%GT = imread("IMG_01_GT.png");

% To visualise the ground truth image, you can
% use the following code.
%L_GT = label2rgb(rgb2gray(GT), 'prism','k','shuffle');
%figure, imshow(L_GT)