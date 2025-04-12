function [labeled_array, knn_model] = classify_3D(input_array, labels_ves, labels_gm, endflag, varargin)
% Function returns a labeled array of vessels (0 or 1), takes in input of
% original array and an array of labels containing the 3D coordinates of
% vessels, and grey matter

% Default values for optional arguments
wave_transform = true;
model = [];

% Parse optional arguments
if ~isempty(varargin)
    for i = 1:2:numel(varargin)
        option_name = varargin{i};
        option_value = varargin{i+1};
        switch option_name
            case 'WaveTransform'
                wave_transform = option_value;
            case 'Model'
                model = option_value;
            otherwise
                error('Unknown option: %s', option_name);
        end
    end
end

if wave_transform
    % Starting off by just using the wavelet transform to construct
    % features, could add in vesselness as well
    
    % Here we are doing a 2D cwt in each direction, this could also be done
    % through a 3D discrete wave transform (waverec3) but the
    % reconstruction of the data to match pixel values makes it more
    % computationally expensive

    disp('Using wave transform for creating features.')
    wavescale = [1 2 5 10]; 
    n = length(wavescale);
    
    wavetype = 'mexh';
    x = zeros(size(input_array,1), size(input_array,2), size(input_array,3), 3*n+1);
    
    % Store original pixel values in the first index of the 4D array
    x(:,:,:,1) = input_array;
    
    for i = 1:size(input_array,3)
        w = cwtft2(input_array(:,:,i),'wavelet',wavetype,'scales',wavescale); % 2-D wavelet transform
        y = squeeze(w.cfs);
        y = reshape(y, size(y,1), size(y,2), 1, size(y,3));
        x(:,:,i,2:2+n-1) = y; % Store wavelet coefficients in the 4D array
    end
    
    % Perform wavelet transform in the x-plane
    for i = 1:size(input_array,1)
        w = cwtft2(squeeze(input_array(i,:,:)),'wavelet',wavetype,'scales',wavescale); % 2-D wavelet transform
        y = squeeze(w.cfs);
        y = reshape(y, size(y,1), size(y,2), 1, size(y,3));
        x(i,:,:,2+n:2+2*n-1) = y; % Store wavelet coefficients in the 4D array
    end
    
    % Perform wavelet transform in the y-plane
    for i = 1:size(input_array,2)
        w = cwtft2(squeeze(input_array(:,i,:)),'wavelet',wavetype,'scales',wavescale); % 2-D wavelet transform
        y = squeeze(w.cfs);
        y = reshape(y, size(y,1), size(y,2), 1, size(y,3));
        x(:,i,:,2+2*n:2+3*n-1) = y; % Store wavelet coefficients in the 4D array
    end

elseif ~wave_transform
    disp('Using Jerman vesselness for creating features.')

    % Adjust sigmas to include multiple values
    sigmas = [5, 10, 15]; % Example vector of sigma values
    spacing = [2.83; 2.83; 4.99]; % Assuming isotropic spacing. Adjust if your image has different spacings.
    tau = 0.1; % Controls response uniformity. Adjust according to your needs.
    brightondark = true; % Set based on whether vessels are bright on a dark background.
    
    % Preallocate for vesselness results; adjust depth based on number of sigmas
    % The first layer is the original image, followed by a layer for each sigma
    x = zeros(size(input_array,1), size(input_array,2), size(input_array,3), numel(sigmas) + 1);
    %x(:,:,:,1) = input_array; % Original image in the first layer
    
    % Apply Jerman vesselness filter for each sigma
    for i = 1:length(sigmas)
        sigma = sigmas(i);
        ves_result = vesselness3D(input_array, sigma, spacing, tau, brightondark);
        % Store the vesselness result for this sigma
        x(:,:,:,i) = ves_result; % Store in subsequent layers of x
    end

if endflag == 0
    % Extract features for positive labels
    pos_features = zeros(size(labels_ves, 1), size(x, 4));
    for idx = 1:size(labels_ves, 1)
        x_coord = labels_ves(idx, 1);
        y_coord = labels_ves(idx, 2);
        z_coord = labels_ves(idx, 3);
        pos_features(idx, :) = squeeze(x(x_coord, y_coord, z_coord, :));
    end
    
    % Extract features for negative labels
    neg_features = zeros(size(labels_gm, 1), size(x, 4));
    for idx = 1:size(labels_gm, 1)
        x_coord = labels_gm(idx, 1);
        y_coord = labels_gm(idx, 2);
        z_coord = labels_gm(idx, 3);
        neg_features(idx, :) = squeeze(x(x_coord, y_coord, z_coord, :));
    end
    
    % Create labels for positive and negative samples
    pos_labels = ones(size(labels_ves, 1), 1);
    neg_labels = zeros(size(labels_gm, 1), 1);
    
    % Concatenate positive and negative features and labels
    all_features = [pos_features; neg_features];
    all_labels = [pos_labels; neg_labels];
    
    % Combine features and labels into a single matrix
    combined_data = [all_features, all_labels];
    
    % Shuffle the combined data
    shuffled_data = combined_data(randperm(size(combined_data, 1)), :);
    
    % Separate shuffled features and labels
    features = shuffled_data(:, 1:end-1);
    labels = shuffled_data(:, end);
end


% Train kNN model
if isempty(model)
    % Train kNN model
    k = 5; % Number of neighbors
    knn_model = fitcknn(features, labels, 'NumNeighbors', k);
else
    % If model was provided as an argument to the function
    knn_model = model;
end

% Flatten the first three dimensions of x into a single dimension
flattened_size = size(input_array, 1) * size(input_array, 2) * size(input_array, 3);
% Reshape x to match the structure of all_voxel_features
reshaped_x = reshape(x, flattened_size, []);

% Directly assign reshaped_x to all_voxel_features
all_voxel_features = reshaped_x;

% Predict in batches
predictions = predict(knn_model, all_voxel_features);

% Reshape predictions to match the input_array dimensions
labeled_array = reshape(predictions, size(input_array));  
end