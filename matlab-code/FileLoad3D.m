% File loading and compression

% Initialize variables
dataPath = '';
fileList = [];

%% Directory Selection
% Loop until valid folder is selected or user decides to exit
while isempty(fileList)
    % Prompt user to select a folder containing TIFF files
    dataPath = uigetdir('', 'Select a folder containing TIFF files');
    if dataPath == 0
        error('No folder selected for TIFF files. Exiting script.');
    end

    % Get files
    fileList = dir(fullfile(dataPath, '*.tif'));

    % Check if there are any .tif files in the directory
    if isempty(fileList)
        choice = questdlg('No TIFF files found in the selected directory. Do you want to try another folder?', ...
            'No TIFF files', 'Yes', 'No', 'Yes');
        if strcmp(choice, 'No')
            error('User opted to exit. No TIFF files found.');
        end
    end
end

% Prompt user to select a save directory
savePath = uigetdir('', 'Select a folder to save the output MAT files');
if savePath == 0
    error('No folder selected for saving MAT files. Exiting script.');
end

%% File Loading

% Get the dimensions of the images
info = imfinfo(fullfile(dataPath, fileList(1).name));
numRows = info(1).Height;
numCols = info(1).Width;
numSlices = numel(fileList);

%% Volume Sectioning, Filtering, and Saving

% Define a chunk size
targetChunkSize = [1000 1000 1000];

% Calculate the number of chunks in each dimension
numChunks = ceil([numRows, numCols, numSlices] ./ targetChunkSize);

% Adjust chunk size to make them approximately equal in each dimension
actualChunkSize = [ceil(numRows / numChunks(1)), ...
    ceil(numCols / numChunks(2)), ceil(numSlices / numChunks(3))];

% Iterate over the chunks
for i = 1:numChunks(1)
    for j = 1:numChunks(2)
        for k = 1:numChunks(3)

            % Compute starting and ending indices for each chunk
            startRow = (i - 1) * actualChunkSize(1) + 1;
            endRow = min(i * actualChunkSize(1), numRows);
            startCol = (j - 1) * actualChunkSize(2) + 1;
            endCol = min(j * actualChunkSize(2), numCols);
            startSlice = (k - 1) * actualChunkSize(3) + 1;
            endSlice = min(k * actualChunkSize(3), numSlices);

            % Initialize an empty array to store sectioned 3D volumes
            sectMicroscImages = zeros(endRow-startRow + 1, endCol-startCol + 1, endSlice-startSlice + 1, 'uint8');

            % Initialize waitbar
            h = waitbar(0, sprintf('Processing: Row %d/%d, Col %d/%d, Depth %d/%d...', ...
                i, numChunks(1), j, numChunks(2), k, numChunks(3)));

            for r = startSlice:endSlice
                % Load in file and section
                filePath = fullfile(dataPath, fileList(r).name);
                img = imread(filePath);
                sectImg = img(startRow:endRow, startCol:endCol);

                % Store in array
                sectMicroscImages(:, :, r-startSlice+1) = im2uint8(sectImg);

                % Update waitbar
                waitbar((r-startSlice+1)/(endSlice-startSlice), h, ...
                    sprintf('Processing: Row %d/%d, Col %d/%d, Depth %d/%d ... %d%%', ...
                    i, numChunks(1), j, numChunks(2), k, numChunks(3), round((r-startSlice+1)/(endSlice-startSlice)*100)));
            end

            % Close waitbar
            close(h);

            % Adjust image contrast
            sectMicroscImages = imadjustn(sectMicroscImages);

            % Save the sectioned and filtered images to a MAT file
            [folderPath, dataFolder] = fileparts(dataPath);
            saveFolderPath = fullfile(savePath, [dataFolder '_Partitioned']);  % Append '_Partitioned' to the folder name
            if ~exist(saveFolderPath, 'dir')
                mkdir(saveFolderPath);  % Create the folder if it doesn't exist
            end
            fprintf('Saving: Row %d/%d, Col %d/%d, Depth %d/%d...\n', i, numChunks(1), j, numChunks(2), k, numChunks(3));
            save(fullfile(saveFolderPath, ...
                sprintf('%s_Row%d_Col%d_Depth%d.mat', dataFolder, i, j, k)), ...
                'sectMicroscImages', '-v7.3');

            % Display a message indicating the data has been saved
            disp('Save complete');

        end
        % Clear array
        clear sectMicroscImages;
    end
end
