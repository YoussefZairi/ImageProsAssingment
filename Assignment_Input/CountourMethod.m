clear; close all;

% Task 5: Robust method --------------------------

% Load the new image
IMG = imread('IMG_08.png');

% Convert image to grayscale
I_gray = rgb2gray(IMG);

% Rescale image
height = 512;
[rows, cols] = size(I_gray);
new_width = round(cols * (height / rows));
I_resized = imresize(I_gray, [height, new_width]);

% Adaptive Histogram Equalization
I_enhanced = adapthisteq(I_gray, 'NumTiles', [32 32], 'ClipLimit', 0.01);

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

% Step 5: Label Objects
% Separate blood cells and bacteria using size and shape
blood_cell_mask = bwareaopen(imopen(logical(I_cleaned), strel('disk', 14)), 2500);
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
imshow(segmented);
title('Segmented Image');

subplot(2, 2, 4);
imshow(I_filled);
title('Filled');

% Task 6: Performance evaluation -----------------
% Step 1: Load ground truth data
GT = imread("IMG_08_GT.png");
L_GT = label2rgb(rgb2gray(GT), 'prism', 'k', 'shuffle');

%Step 2: Resizing GT
GT_resized = imresize(GT,[size(colored_final, 1), size(colored_final,2)]);

%Step 3: Convert GT to binary masks
blood_cell_mask_GT = GT_resized == 1;
bacteria_mask_GT = GT_resized == 2;

%Step 4: Function to calculate Dice Score
function dice = calculateDice(A,B)
    intersection = nnz(A & B);
    dice = 2* intersection / (nnz(A) + nnz(B));
end

%Step 5: Function to calculate Precision
function precision = calculatePrecision(A, B)
    true_positives = nnz(A & B);
    predicted_positives = nnz(B);
    precision = true_positives / predicted_positives;
end

%Step 6: Function to calculate Recall
function recall = calculateRecall(A, B)
    true_positives = nnz(A & B);
    actual_positives = nnz(A);
    recall = true_positives / actual_positives;
end

%Step 7: Calculate metrics for blood cells
dice_blood_cells = calculateDice(blood_cell_mask_GT, blood_cell_mask);
precision_blood_cells = calculatePrecision(blood_cell_mask_GT, blood_cell_mask);
recall_blood_cells = calculateRecall(blood_cell_mask_GT, blood_cell_mask);
%Step 8: Display blood cells results
fprintf('Blood Cells:\n');
fprintf('Dice Score: %.4f\n', dice_blood_cells);
fprintf('Precision: %.4f\n', precision_blood_cells);
fprintf('Recall: %.4f\n', recall_blood_cells);

%Step 9: Calculate metrics for bacteria
dice_bacteria = calculateDice(bacteria_mask_GT, bacteria_mask);
precision_bacteria = calculatePrecision(bacteria_mask_GT, bacteria_mask);
recall_bacteria = calculateRecall(bacteria_mask_GT, bacteria_mask);
%Step 10: Display bacteria results
fprintf('Bacteria:\n');
fprintf('Dice Score: %.4f\n', dice_bacteria);
fprintf('Precision: %.4f\n', precision_bacteria);
fprintf('Recall: %.4f\n', recall_bacteria);

subplot(2,1,1);
imshow(L_GT);
title('Ground Truth')
subplot(2,1,2);
imshow(colored_final);
title('Colored Final');