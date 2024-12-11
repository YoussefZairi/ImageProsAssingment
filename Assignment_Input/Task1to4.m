clear; close all;

% Task 1: Pre-processing -----------------------
% Step-1: Load input image
I = imread('IMG_01.png');
figure, imshow(I)
title('Original IMG 1')

% Step-2: Covert image to grayscale
I_gray = rgb2gray(I);
figure, imshow(I_gray);
title('Grayscale Image');

% Step-3: Rescale image
height = 512;
[rows,cols] = size(I_gray);
new_width = round(cols * (height / rows));
I_resized = imresize(I_gray, [height, new_width]);

figure, imshow(I_resized);
title('Resized Image');

% Step-4: Produce histogram before enhancing
%4.1 Create Number of Bins % histogram data using number of bins
num_bins = 64;
histogram_data = imhist(I_resized, num_bins);
%4.2 Display the histogram and Add title / labels
figure, bar(histogram_data);
title('Histogram Data Pre Enhance');
xlabel('Intensity Values');
ylabel('Frequency');
% Step-5: Enhance image before binarisation
%Contrast Adjustment for enhancement
I_enhanced = imadjust(I_resized);

figure, imshow(I_enhanced);
title('Enhanced Image (Contrast)')

% Step-6: Histogram after enhancement
%6.1 Create Number of Bins % histogram data using number of bins
num_bins = 64;
histogram_data_e = imhist(I_enhanced, num_bins);
%6.2 Display the histogram and Add title / labels
figure, bar(histogram_data_e);
title('Histogram Data Post Enhance');
xlabel('Intensity Values');
ylabel('Frequency');
% Step-7: Image Binarisation
%7.1 Determine threshold with Otsu's method
threshold = graythresh(I_enhanced);

%7.2 Convert to Binary
I_binary = imbinarize(I_enhanced, threshold);

figure, imshow(I_binary);
title ('Binarized Image Otsu Method');

%7.3 Expermiment with other threshholding methods
%Manual Tresholding:
I_binary_manual = imbinarize(I_enhanced, 0.25);
figure, imshow(I_binary_manual);
title ('Binarized 0.5');
%figure, imshow(I_binary_manual);
%title ('Binarized Manual Test 1');

I_binary_manual2 = imbinarize(I_enhanced, 0.75);

%figure, imshow(I_binary_manual2);
%title('Binarized Manual Test 2');

%Need to find a way to store all files locally stored in directory!!! with
%github to make looking through easy


% Task 2: Edge detection ------------------------
%Sobel Edge Detection
sobel_edges = edge(I_binary, 'Sobel');

%figure, imshow(sobel_edges);
%title('Sobel Edge Detection')

%Prewitt Edge Detecttion
prewitt_edges = edge(I_binary, 'prewitt');

%figure, imshow(prewitt_edges);
%title('Prewitt Edge Detection');

%Canny Edge Detection
canny_edges = edge(I_binary, 'Canny', 0.86);

figure, imshow(canny_edges);
title('Canny Edge Detection')
imwrite(canny_edges, 'edges_canny.png');
%Roberts Edge Detection
roberts_edges = edge(I_binary, 'Roberts');

%figure, imshow(roberts_edges);
%title('Roberts Edge Detection');

figure;
subplot(2, 2, 1);
imshow(sobel_edges);
title('Sobel Edge Detection');

subplot(2, 2, 2);
imshow(prewitt_edges);
title('Prewitt Edge Detection');

subplot(2, 2, 3);
imshow(canny_edges);
title('Canny Edge Detection');

subplot(2, 2, 4);
imshow(roberts_edges);
title('Roberts Edge Detection');
%After reveiwing all the edge detection, Canny seems to produce the highest
%quality edge detection!! FOR REPORT!
% Save the selected edge detection output (e.g., Canny)
imwrite(roberts_edges, 'edges_roberts.png');

% Task 3: Simple segmentation --------------------
I_edge = imread('edges_roberts.png');
% Add padding around the image to include boundary cells
%After testing I realized some of the corner cells were not being detected
padded_edges = padarray(I_edge, [1, 1], 0, 'both');
se = strel('disk', 1);
a_dilated_edges = imdilate(padded_edges, se);

figure, imshow(a_dilated_edges);
title('Dilated Edge Image');

%Bridging the padded edge image
dilated_edges = bwmorph(a_dilated_edges,"bridge");

%https://blogs.mathworks.com/steve/2013/09/05/defining-and-filling-holes-on-the-border-of-an-image/
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


I_filled = d_edges_a_filled | d_edges_b_filled | d_edges_c_filled | d_edges_d_filled;
figure, imshow(I_filled);
title('Filled Image')

I_cleaned = bwareaopen(I_filled, 50);
I_smoothed = imclose(I_cleaned, strel('disk',2));

I_labeled = bwlabel(I_smoothed);

figure, imshow(label2rgb(I_labeled, 'jet','k'));
title('Labeled Objects')

%filter based on properties
stats = regionprops(I_labeled, 'Area', 'Eccentricity');
blood_cells = ismember(I_labeled, find([stats.Area] > 2000  ));
bacteria = ismember(I_labeled, find([stats.Area] <=2000 & [stats.Area] > 500 ));
final_binary = blood_cells | bacteria;

%My Labeled image shows the bacteria and blood cells fina dn so I decided I
%will study the area and eccentricity of both objects to determine the
%stats of blood cells and bacteria above, 

for i = 1:numel(stats)
    fprintf('Object %d: Area = %.2f, Eccentricity = %.2f\n', i, stats(i).Area, stats(i).Eccentricity);
end

figure, imshow(final_binary);
title('Segmented Blood Cells or Bacteria');

%To ensure my classification of objects is correct test to see blood cells
%and bacteria
% Visualize blood cells
%figure, imshow(blood_cells);
%title('Blood Cells');

% Visualize bacteria
%figure, imshow(bacteria);
%title('Bacteria');

%Save Segmented Image
imwrite(final_binary, 'segmented_image.png');
% -------------------- ------------------------------------------------------------
% Task 4: Object Recognition --------------------

% Load the new image
IMG_11 = imread('IMG_11.png');

%Covert image to grayscale
I_gray = rgb2gray(IMG_11);
figure, imshow(I_gray);
title('Grayscale Image');

%Rescale image
height = 512;
[rows,cols] = size(I_gray);
new_width = round(cols * (height / rows));
I_resized = imresize(I_gray, [height, new_width]);

%Enhance Image
I_enhanced = imadjust(I_resized);

%Determine threshold with Otsu's method
threshold = graythresh(I_enhanced);

%Convert to Binary
I_binary = imbinarize(I_enhanced, threshold);

%Edge Detection
I_edge = edge(I_binary, 'roberts');

%Add Padding
padded_edges = padarray(I_edge, [1, 1], 0, 'both');
se = strel('disk', 1);

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

% Remove noise
I_cleaned = bwareaopen(I_filled, 500);

% Smooth edges
I_smoothed = imclose(I_cleaned, strel('disk', 2));

% Label connected components
I_labeled = bwlabel(I_smoothed);

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
blood_cells_mask = ismember(I_labeled, find([stats.Area] > 2000 & circularities > 0.50 ));
bacteria_mask = ismember(I_labeled, find([stats.Area] <=2000 & [stats.Area] & circularities < 0.50 ));

% Create an RGB image
colored_final = zeros([size(I_labeled), 3]);
colored_final(:,:,1) = blood_cells_mask;
colored_final(:,:,2) = bacteria_mask;
colored_final(:,:,3) = bacteria_mask;
figure, imshow(colored_final);
title('Colored Final')