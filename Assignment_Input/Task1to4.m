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

% Task 3: Simple segmentation --------------------

% Task 4: Object Recognition --------------------