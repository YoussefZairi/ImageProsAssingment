% Create the output directory for that image
output_dir = 'IMG15 Output';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Task 5: Robust method --------------------------

% Load the new image
IMG = imread('IMG_15.png');

% Convert image to grayscale
I_gray = rgb2gray(IMG);

% Rescale image
height = 512;
[rows, cols] = size(I_gray);
new_width = round(cols * (height / rows));
I_resized = imresize(I_gray, [height, new_width]);

% Save the resized grayscale image
imwrite(I_resized, fullfile(output_dir, '01_Resized_Gray.png'));

% Adaptive Histogram Equalization
I_enhanced = adapthisteq(I_gray, 'NumTiles', [32 32], 'ClipLimit', 0.01);
imwrite(I_enhanced, fullfile(output_dir, '02_Enhanced_Image.png'));  % Save enhanced image

% Apply Gaussian smoothing
I_smoothed = imgaussfilt(I_enhanced, 2);  % Gaussian filter with sigma=2
imwrite(I_smoothed, fullfile(output_dir, '03_Smoothed_Image.png'));  % Save smoothed image

gamma = 0.8;  % Gamma < 1 brightens, Gamma > 1 darkens
I_gamma = imadjust(I_smoothed, [], [], gamma);

% Normalize to range [0, 1]
I_normalized = mat2gray(I_gamma);
imwrite(I_normalized, fullfile(output_dir, '04_Normalized_Image.png'));  % Save normalized image

% Otsu's Method
thresh = graythresh(I_normalized); 
I_binary = imbinarize(I_normalized, thresh);
imwrite(I_binary, fullfile(output_dir, '05_Binary_Image.png'));  % Save binary image

% Step 1: Edge Detection
I_edge = edge(I_binary, 'roberts'); % Detect edges
imwrite(I_edge, fullfile(output_dir, '06_Edge_Detection.png'));  % Save edge detection image

% Step 2: Morphological Processing
se_close = strel('disk', 3);
I_closed = imclose(I_edge, se_close); % Close small gaps
I_bridged1 = bwmorph(I_closed, "bridge"); % Bridge gaps in edges
imwrite(I_bridged1, fullfile(output_dir, '07_Bridged_Edges.png'));  % Save bridged edges

% Step 3: Segmentation using Active Contour
mask = zeros(size(I_binary)); 
mask(50:end-50, 50:end-50) = 1;  % Initial mask (avoid borders)
bw = activecontour(I_binary, mask, 1000); % Segment with active contour
segmented = bwareaopen(bw, 700); % Remove small objects (<700 pixels)
imwrite(segmented, fullfile(output_dir, '08_Segmented_Image.png'));  % Save segmented image

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
imwrite(I_filled, fullfile(output_dir, '09_Filled_Image.png'));  % Save filled image

% Remove very small objects
I_cleaned = bwareaopen(I_filled, 400);
imwrite(I_cleaned, fullfile(output_dir, '10_Cleaned_Image.png'));  % Save cleaned image

% Step 5: Label Objects
% Separate blood cells and bacteria using size and shape
blood_cell_mask = bwareaopen(imopen(logical(I_cleaned), strel('disk', 14)), 2500);
imwrite(blood_cell_mask, fullfile(output_dir, '11_Blood_Cell_Mask.png'));  % Save blood cell mask

bacteria_mask = I_cleaned & ~blood_cell_mask;
bacteria_mask = bwareaopen(bacteria_mask, 300);
imwrite(bacteria_mask, fullfile(output_dir, '12_Bacteria_Mask.png'));  % Save bacteria mask

% Combine labeled masks into an RGB image
colored_final = zeros([size(I_cleaned), 3]);
colored_final(:,:,1) = blood_cell_mask; % Red for blood cells
colored_final(:,:,2) = bacteria_mask;   % Green for bacteria
colored_final(:,:,3) = bacteria_mask;   % Blue for bacteria to create cyan
imwrite(colored_final, fullfile(output_dir, '13_Colored_Final.png'));  % Save final colored image
