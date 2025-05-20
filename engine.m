% Image Search Engine Using Visual Content Matching in MATLAB

% Step 1: Load Image Database
imageFolder = 'image_database'; 
imds = imageDatastore(imageFolder, 'FileExtensions', {'.jpg', '.png'});

% Step 2: Feature Extraction and Database Construction
featuresDB = [];
imageIDs = [];

for i = 1:numel(imds.Files)
    img = imread(imds.Files{i});
    grayImg = rgb2gray(img);

    % SURF Feature Extraction
    points = detectSURFFeatures(grayImg);
    [features, ~] = extractFeatures(grayImg, points);

    featuresDB = [featuresDB; features];
    imageIDs = [imageIDs; i * ones(size(features, 1), 1)];
end

% Step 3: Build KD-Tree for Efficient Search
kdtree = KDTreeSearcher(featuresDB);

% Step 4: Query Image Processing
queryImage = imread('query.jpg');
grayQuery = rgb2gray(queryImage);

queryPoints = detectSURFFeatures(grayQuery);
[queryFeatures, ~] = extractFeatures(grayQuery, queryPoints);

% Step 5: KD-Tree Nearest Neighbor Search
[nearestIdx, ~] = knnsearch(kdtree, queryFeatures);

% Step 6: Identify Matched Images
matchedIDs = imageIDs(nearestIdx);

% Step 7: Display Results
topMatches = unique(matchedIDs);

figure;
subplot(1, 2, 1); imshow(queryImage); title('Query Image');
subplot(1, 2, 2); 
imshow(imread(imds.Files{topMatches(1)})); 
title('Top Matched Image');

