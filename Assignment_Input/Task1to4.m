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
I_binary_manual = imbinarize(I_enhanced, 0.5);

figure, imshow(I_binary_manual);
title ('Binarized Manual Test 1');

I_binary_manual2 = imbinarize(I_enhanced, 0.75)

figure, imshow(I_binary_manual2);
title('Binarized Manual Test 2');

%Need to find a way to store all files locally stored in directory!!! with
%github to make looking through easy


% Task 2: Edge detection ------------------------
%Sobel Edge Detection
sobel_edges = edge(I_enhanced, 'Sobel');

figure, imshow(sobel_edges);
title('Sobel Edge Detection')
%Prewitt Edge Detecttion
prewitt_edges = edge(I_enhanced, 'prewitt');

figure, imshow(prewitt_edges);
title('Prewitt Edge Detection');
%Canny Edge Detection
canny_edges = edge(I_enhanced, 'Canny');

figure, imshow(canny_edges);
title('Canny Edge Detection')
%Roberts Edge Detection
roberts_edges = edge(I_enhanced, 'Roberts');

figure, imshow(roberts_edges);
title('Roberts Edge Detection');

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
imwrite(canny_edges, 'edges_canny.png');

% Task 3: Simple segmentation --------------------
I_edge = imread('edges_canny.png');
se = strel('disk', 2);
dilated_edges = imdilate(I_edge, se);

figure, imshow(dilated_edges);
title('Enhanced Edge Image');

I_filled = imfill(dilated_edges, 'holes');

figure, imshow(I_filled);
title('Filled Image')

I_cleaned = bwareaopen(I_filled, 50);
I_smoothed = imclose(I_cleaned, strel('disk',3));

I_labeled = bwlabel(I_smoothed);

figure, imshow(label2rgb(I_labeled, 'jet','k'));
title('Labeled Objects')

%filter based on properties
stats = regionprops(I_labeled, 'Area', 'Eccentricity');
final_binary = ismember(I_labeled, find([stats.Area] > 50 & [stats.Eccentricity] < 0.8));

figure, imshow(final_binary);
title('Segmented Blood Cells or Bacteria');

%Save Segmented Image
imwrite(final_binary, 'segmented_image.png');
% Task 4: Object Recognition --------------------