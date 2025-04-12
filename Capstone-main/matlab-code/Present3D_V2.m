%% Initialization
clearvars('fig', 'data');
close all
addpath(genpath(fullfile(pwd, 'Jerman'))); % Make sure the Jerman vesselness package folder is in the same folder as this program

% Prompt the user to select a .mat file
[fileName, pathName] = uigetfile('*.mat', 'Select a .mat file');
if isequal(fileName,0)
    disp('User selected Cancel');
    return;
else
    disp(['User selected ', fullfile(pathName, fileName)]);
    data.matFileName = fileName; % Store the selected file name for later use
    data.matPathName = pathName;
    
    % Load the file
    fileContents = load(fullfile(pathName, fileName));
    
    % Look for a 3D array among the variables in the loaded file
    vars = fieldnames(fileContents);
    found = false;
    for i = 1:length(vars)
        varName = vars{i};
        tempVar = fileContents.(varName);
        if ndims(tempVar) == 3 && isnumeric(tempVar)
            sectMicroscImages = tempVar;
            found = true;
            disp(['Loaded 3D array: ', varName]);
            break;
        end
    end
    
    if ~found
        error('No suitable 3D array found in the file.');
    end
end

%% PARAMETERS
subarraydim = 300; % size of sample cube for quick visulisation, adjust for speed and to not overfill RAM

data.classification_model = 'knn'; % select classification method for points collected, 
                                   % currently either 'knn','naivebayes' or 'logisticregression'

data.chunkSize = 275; % for later extracting chunks for RAM solving

data.overlap = 25; % amount of overlap to use for the chunks to avoid edge artefacts

% Jerman Parameters
% Adjust sigmas to include multiple values for range of possible vessel sizes
data.sigmas = [5, 10, 15]; % vector of sigma values
data.tau = 0.1; % Controls response uniformity. Adjust according to your needs.
data.brightondark = true; % Set based on whether vessels are bright on a dark background.
data.spacing = [2.83; 2.83; 4.99]; % Pixel sizes for data, adjust accordingly

%%
% Assuming 'sectMicroscImages' is loaded and available now
data.fullarray = sectMicroscImages;
data.fullarray = imadjustn(data.fullarray); %contrast adjust

% Original dimensions
[data.X, data.Y, data.Z] = size(data.fullarray);

% Calculate padding needed to make dimensions divisible by chunkSize
% Check if dimension is less than chunkSize or divisible by chunkSize, set pad to 0 in these cases
padX = data.chunkSize - mod(data.X, data.chunkSize);
padY = data.chunkSize - mod(data.Y, data.chunkSize);
padZ = data.chunkSize - mod(data.Z, data.chunkSize);

padX = padX * (padX ~= data.chunkSize) * (data.X >= data.chunkSize);
padY = padY * (padY ~= data.chunkSize) * (data.Y >= data.chunkSize);
padZ = padZ * (padZ ~= data.chunkSize) * (data.Z >= data.chunkSize);

% Store the adjusted padding
data.padX = padX;
data.padY = padY;
data.padZ = padZ;

% Pad the array
data.fullarray = padarray(data.fullarray, [padX, padY, padZ], 'post');

% Now data.fullarray is padded instead of trimmed, and its dimensions are divisible by chunkSize

% Calculate the starting indices for the sub-array
startX = round((data.X - min(data.X, subarraydim)) / 2);
startY = round((data.Y - min(data.Y, subarraydim)) / 2);
startZ = round((data.Z - min(data.Z, subarraydim)) / 2);

% Adjust the start indices to ensure they are positive and make sense
startX = max(startX, 1);
startY = max(startY, 1);
startZ = max(startZ, 1);

% Calculate the end indices for the sub-array, ensuring they do not exceed the original dimensions
endX = startX + min(data.X, subarraydim) - 1;
endY = startY + min(data.Y, subarraydim) - 1;
endZ = startZ + min(data.Z, subarraydim) - 1;

% Extract the sub-array with adjusted dimensions
arrayChoice = data.fullarray(startX:endX, startY:endY, startZ:endZ);

%% Scroll/Point Code
% Assuming imgData is your 3D image data of type uint8
data.dataArray = arrayChoice;
data.fig2 = [];

data.pointsList_gm = []; % Separate for gm in all planes
data.pointsList_ves = []; % Separate for ves in all planes
data.lastSelectedPoint = [];

% Create a figure window with a specified size
screenSize = get(0, 'ScreenSize'); % Get screen size to make the figure window appropriately sized
figWidth = screenSize(3) * 0.4; % Half of the screen width
figHeight = screenSize(4) * 0.8; % 80% of the screen height
fig = figure('Name', 'Slice Viewer', 'NumberTitle', 'off', ...
             'Position', [1, screenSize(4)*0.1, figWidth, figHeight]);
colormap jet

% Variable to store the selected plane
data.selectedPlane = 3; % Default to 'xy' plane

data.sliceNumber = 1; % Start with the first slice
data.ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0.1 0.2 0.8 0.7]);
displaySlice(data.sliceNumber,data,fig);

% Adjust the slider to be positioned relative to the figure size
data.slider = uicontrol('Parent', fig, 'Style', 'slider', 'Units', 'normalized', ...
                   'Position', [0.1, 0.03, 0.8, 0.05], 'Value', 1, 'Min', 1, ...
                   'Max', size(data.dataArray, data.selectedPlane), 'SliderStep', ...
                   [1/(size(data.dataArray, data.selectedPlane)-1) , ...
                   10/(size(data.dataArray, data.selectedPlane)-1)]);

% Display the 'spacing' list under the scroll bar
data.spacingText = uicontrol('Parent', fig, 'Style', 'text', 'Units', 'normalized', ...
                             'Position', [0.1, 0.02, 0.8, 0.05], 'String', ...
                             sprintf('Pixel Spacing: X=%.2f, Y=%.2f, Z=%.2f', data.spacing), ...
                             'HorizontalAlignment', 'left');

set(data.spacingText, 'FontSize', 11);

% Button for sample 3D segmentation
btn1 = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Sample 3D Volume Segment', ...
                'Units', 'normalized', 'Position', [0.02, 0.95, 0.22, 0.04], ...
                'Callback', @test_classification);

% Button for Full 3D segmentation
btn2 = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'FULL 3D Volume Segment', ...
                'Units', 'normalized', 'Position', [0.26, 0.95, 0.22, 0.04], ...
                'Callback', @full_classification_choice);

% Radio buttons for plane selection
data.radio_xy = uicontrol('Parent', fig, 'Style', 'radiobutton', 'String', 'xy', ...
                          'Units', 'normalized', 'Position', [0.6, 0.95, 0.1, 0.04], ...
                          'Callback', @selectPlane, 'Value', 1);

data.radio_xz = uicontrol('Parent', fig, 'Style', 'radiobutton', 'String', 'yz', ...
                          'Units', 'normalized', 'Position', [0.7, 0.95, 0.1, 0.04], ...
                          'Callback', @selectPlane);

data.radio_yz = uicontrol('Parent', fig, 'Style', 'radiobutton', 'String', 'xz', ...
                          'Units', 'normalized', 'Position', [0.8, 0.95, 0.1, 0.04], ...
                          'Callback', @selectPlane);

% Button to clear points from the current slice
btnClear = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Clear Slice Points', ...
                     'Units', 'normalized', 'Position', [0.65, 0.04, 0.22, 0.04], ...
                     'Callback', @clearPoints);

% Adjust the slider position slightly to make room for the Clear Points button
set(data.slider, 'Position', [0.1, 0.09, 0.8, 0.05]);

% Update the listener for the slider
guidata(fig, data); % Save changes to 'data'
addlistener(data.slider, 'Value', 'PostSet', @(s,e) updateSlice(data,fig));
guidata(fig, data); % Save changes to 'data'


%% Scroll Functions
function updateSlice(~,~)
    fig = getFigureHandleByName('Slice Viewer'); % Get the current figure handle
    data = guidata(fig); % Retrieve the data structure

    % Get the rounded slider value to avoid fractional slice numbers
    sliceNum = round(data.slider.Value);
    data.sliceNumber = sliceNum;
    guidata(fig, data); % Save changes to 'data'
    displaySlice(data.sliceNumber,data,fig); % Updated function call
    guidata(fig, data); % Save changes to 'data'
end

function updateSlider(~,~)
    fig = getFigureHandleByName('Slice Viewer'); % Get the current figure handle
    data = guidata(fig); % Retrieve the data structure
    % Determine the new slice number based on the selected plane and last selected point
    if ~isempty(data.lastSelectedPoint)
        switch data.selectedPlane
            case 2 % yz plane
                newSliceNumber = data.lastSelectedPoint(1);
            case 1 % xz plane
                newSliceNumber = data.lastSelectedPoint(2);
            case 3 % xy plane
                newSliceNumber = data.lastSelectedPoint(3);
        end
    else
        newSliceNumber = 1; % Default value if no point has been selected
    end

    % Update the slider properties
    set(data.slider, 'Min', 1, 'Max', size(data.dataArray, data.selectedPlane), 'Value', newSliceNumber, 'SliderStep', [1/(size(data.dataArray, data.selectedPlane)-1) , 10/(size(data.dataArray, data.selectedPlane)-1)]);
    data.sliceNumber = newSliceNumber; % Update the slice number
    guidata(fig, data); % Save changes to 'data'
end

function selectPlane(source, ~)
    fig = source.Parent; % Assuming this callback is directly associated with the figure
    data = guidata(fig); % Retrieve the current GUI data

    % Update the selected plane based on the radio button selection
    switch source.String
        case 'xy'
            data.selectedPlane = 3;
            set(data.radio_xz, 'Value', 0);
            set(data.radio_yz, 'Value', 0);
            guidata(fig, data); % Update data structure
        case 'yz'
            data.selectedPlane = 1;
            set(data.radio_xy, 'Value', 0);
            set(data.radio_yz, 'Value', 0);
            guidata(fig, data); % Update data structure
        case 'xz'
            data.selectedPlane = 2;
            set(data.radio_xy, 'Value', 0);
            set(data.radio_xz, 'Value', 0);
            guidata(fig, data); % Update data structure
    end

    % Immediately reflect change in displayed slice if a point has been selected
    if ~isempty(data.lastSelectedPoint)
        switch data.selectedPlane
            case 2 % Switching to yz plane
                data.sliceNumber = data.lastSelectedPoint(1);
            case 1 % Switching to xz plane
                data.sliceNumber = data.lastSelectedPoint(2);
            case 3 % Switching to xy plane
                data.sliceNumber = data.lastSelectedPoint(3);
        end
        set(data.slider, 'Value', data.sliceNumber); % Update the slider position
        guidata(fig, data); % Update data structure
        displaySlice(data.sliceNumber, data, fig);
    end

    % Reset other radio buttons to ensure only one is selected
    set(data.radio_xy, 'Value', (data.selectedPlane == 3));
    set(data.radio_xz, 'Value', (data.selectedPlane == 1));
    set(data.radio_yz, 'Value', (data.selectedPlane == 2));

    % Update the slider and slice display
    updateSlider(data, fig);
    guidata(fig, data); % Update data structure
    drawnow; % Force MATLAB to process the change
    updateSlice(data, fig); % Immediately reflect change
end
function getPoint(~, ~)
    fig = getFigureHandleByName('Slice Viewer'); % Get the current figure handle
    data = guidata(fig); % Retrieve the data structure
    cp = get(data.ax, 'CurrentPoint'); % Get current point in axis coordinates
    x = round(cp(1,1));
    y = round(cp(1,2));

     % Update the last selected point based on the plane
    switch data.selectedPlane
        case 2 % yz plane
            data.lastSelectedPoint = [data.sliceNumber, y, x]; % Note the order (Slice, Y, X)
        case 1 % xz plane
            data.lastSelectedPoint = [y, data.sliceNumber, x]; % Note the order (Y, Slice, X)
        case 3 % xy plane
            data.lastSelectedPoint = [y, x, data.sliceNumber]; % Note the order (X, Y, Slice)
    end
    
    switch data.selectedPlane
        case 2 % yz plane
            coord = [data.sliceNumber, y, x];
            plotY = y; % For yz plane, x from click is mapped to vertical axis in the plot
            plotX = x; % y from click is along the horizontal axis
        case 1 % xz plane
            coord = [y, data.sliceNumber, x];
            plotX = y; % x remains x for horizontal axis
            plotY = x; % y from click is mapped to vertical axis in the plot
        case 3 % xy plane
            coord = [y, x, data.sliceNumber];
            plotX = y; % Direct mapping for horizontal axis
            plotY = x; % Direct mapping for vertical axis
    end

    % Determine the color and list to update based on the click type
    mouseClickType = get(fig, 'SelectionType');
    % Plot the points with adjusted coordinates
    if strcmp(mouseClickType, 'alt') % Right click for grey matter
        data.pointsList_gm = [data.pointsList_gm; coord];
        guidata(fig, data);
        %plot(data.ax, plotX, plotY, 'go', 'MarkerSize', 6, 'LineWidth', 1); % Adjusted plot
    elseif strcmp(mouseClickType, 'normal') % Left click for vessels
        data.pointsList_ves = [data.pointsList_ves; coord];
        guidata(fig, data);
        %plot(data.ax, plotX, plotY, 'ro', 'MarkerSize', 6, 'LineWidth', 1); % Adjusted plot
    end
    guidata(fig, data);
    displaySlice(data.sliceNumber, data, fig); % Refresh display
end

function displaySlice(sliceNum,data,fig)
    % Fetch and reshape slice based on the selected plane
    switch data.selectedPlane
        case 2 % 'yz' plane
            display_image = squeeze(data.dataArray(sliceNum, :, :));
        case 1 % 'xz' plane
            display_image = squeeze(data.dataArray(:, sliceNum, :));
        case 3 % 'xy' plane (default case)
            display_image = squeeze(data.dataArray(:, :, sliceNum));
    end

    % Display the slice
    imgObj = imagesc(display_image, 'Parent', data.ax); 
    axis(data.ax, 'image'); % Maintain aspect ratio
    set(data.ax, 'XLim', [0.5, size(display_image, 2)+0.5], 'YLim', [0.5, size(display_image, 1)+0.5]); % Adjust axis limits
    guidata(fig, data); % Save changes to 'data'
    % Set callbacks for interactive point selection
    set(imgObj, 'ButtonDownFcn', @getPoint);
    set(data.ax, 'HitTest', 'on');
    set(imgObj, 'HitTest', 'on', 'PickableParts', 'all');

    % Update aspect ratio based on the displayed image dimensions
    %[imgHeight, imgWidth] = size(display_image);
    % Optionally, manually set the aspect ratio (uncomment the next line to use it)
    % set(data.ax, 'DataAspectRatio', [1, imgHeight/imgWidth, 1]);
    axis(data.ax, 'image'); % This sets the aspect ratio so that data units are equal in every direction.
    hold(data.ax, 'on');
    
    % Label axes based on the selected plane
    switch data.selectedPlane
        case 1 % 'yz' plane
            xlabel(data.ax, 'Z');
            ylabel(data.ax, 'Y');
            title(data.ax, 'YZ Plane');
        case 2 % 'xz' plane
            xlabel(data.ax, 'Z');
            ylabel(data.ax, 'X');
            title(data.ax, 'XZ Plane');
        case 3 % 'xy' plane
            xlabel(data.ax, 'X');
            ylabel(data.ax, 'Y');
            title(data.ax, 'XY Plane');
    end

    guidata(fig, data); % Save changes to 'data'
    vesPoints = [];
    gmPoints = [];

    gmList = data.pointsList_gm;
    vesList = data.pointsList_ves;

    % Retrieve and filter 3D lists of points, if they exist
    if ~isempty(gmList)

        % Filter points for the current slice and plane
        switch data.selectedPlane
            case 2 % 'yz' plane, x is the slicing dimension
                gmPoints = gmList(gmList(:,1) == sliceNum, [2,3]);
            case 1 % 'xz' plane, y is the slicing dimension
                gmPoints = gmList(gmList(:,2) == sliceNum, [1,3]);
            case 3 % 'xy' plane, z is the slicing dimension
                gmPoints = gmList(gmList(:,3) == sliceNum, [1,2]);
        end

        % Plot grey matter points, if any
        if ~isempty(gmPoints)
            plot(data.ax, gmPoints(:,2), gmPoints(:,1), 'go', 'MarkerSize', 6, 'LineWidth', 1);
        end
    end

    if ~isempty(vesList)
        % Filter points for the current slice and plane
        switch data.selectedPlane
            case 2 % 'yz' plane, x is the slicing dimension
                vesPoints = vesList(vesList(:,1) == sliceNum, [2,3]);
            case 1 % 'xz' plane, y is the slicing dimension
                vesPoints = vesList(vesList(:,2) == sliceNum, [1,3]);
            case 3 % 'xy' plane, z is the slicing dimension
                vesPoints = vesList(vesList(:,3) == sliceNum, [1,2]);
        end

        % Plot vessel points, if any
        if ~isempty(vesPoints)
            plot(data.ax, vesPoints(:,2), vesPoints(:,1), 'ro', 'MarkerSize', 6, 'LineWidth', 1);
        end
    end

    hold(data.ax, 'off');

    %AIDAN 2D Visulisation 
    if ~isempty(vesPoints) && ~isempty(gmPoints)

        I = get(imgObj, 'CData');
        I2 = uint8(repmat(I,[1,1,3]));
        I2(:,:,1) = 255 - I2(:,:,1); % Invert green channel
        I2(:,:,2) = 255 - I2(:,:,2); 
        
        input_array = im2uint16(I);
        % Adjust sigmas to include multiple values
        sigmas = data.sigmas; % vector of sigma values
        spacing = data.spacing; % Adjust if your image has different spacings.
        tau = data.tau; % Controls response uniformity. Adjust according to your needs.
        brightondark = data.brightondark; % Set based on whether vessels are bright on a dark background.
        n = length(sigmas);
        
        x1 = zeros(size(input_array, 1), size(input_array, 2), n+1);
        
        % Store original pixel values in the first index of the 3D array
        x1(:, :, 1) = input_array;
    
        % Apply Jerman vesselness filter for each sigma
        for i = 1:length(sigmas)
            sigma = sigmas(i);
            w = vesselness2D(input_array,sigma, spacing, tau, brightondark); %2-D vesselness filter
            % Store the vesselness result for this sigma
            x1(:, :, i+1) = w;
        end
        
        x_min = min(x1, [], [1 2]); % Find the minimum value along dimensions 1 and 2
        x_max = max(x1, [], [1 2]); % Find the maximum value along dimensions 1 and 2
        
        x_range = x_max - x_min; % Compute the range
        
        % Avoid division by zero by setting x_range to 1 where it equals 0
        x_range(x_range == 0) = 1;
        
        % Perform normalization
        x1 = (x1 - repmat(x_min, [size(x1, 1), size(x1, 2), 1])) ./ repmat(x_range, [size(x1, 1), size(x1, 2), 1]);
        
        [a,b,para_len] = size(x1); %a,b as above; para_len = size(wavescale)
    
        p = reshape(x1,a*b,para_len);%p is 2D; each normalized wavelet transform is in each of the para_len columns

        seed1 = [];
        seed2 = [];
        for i=1:size(vesPoints, 1)
            c = vesPoints(i,2);
            d = vesPoints(i,1);
            seed1 = [seed1 (c-1)*a+d]; % append coordinates to seed1
        end
        for i=1:size(gmPoints, 1)
            c = gmPoints(i,2);
            d = gmPoints(i,1);
            seed2 = [seed2 (c-1)*a+d]; % append coordinates to seed1
        end
        d1 = pdist2(p,p(seed1,:)); % Euclidean distance between the original p data matrix and the row of the p matrix at every mouselick
        d2 = pdist2(p,p(seed2,:));
        pred = sign(min(d1,[],2)-min(d2,[],2)); % subtracts the smallest row value of d1 and d2, then converts the result to a vector with values 1, 0 or -1
    
        figure(2)
        set(gcf, 'position', [580 160 500 550]);
        temp = reshape(pred,[a b]); % re-arranges the vector to an image matrix
        imagesc(temp)
        title('segmented image')
        colormap gray
        axis image
    
        figure(3)
        set(gcf, 'position', [1050 160 500 550]);
        I2(:,:,3) = (1-temp)*255;
        image(I2)
        title({'Considered vessels (blue) and GM (yellow)','superimposed on the original image',...
            'grey to black = increasing chance of blood vessels not labelled',...
            'cyan to white = increasing chance of incorrectly labelled vessel'}) 
        axis image
    else
        % Close figure 2 if it exists
        if isgraphics(2, 'figure')
            close(2);
        end
        % Close figure 3 if it exists
        if isgraphics(3, 'figure')
            close(3);
        end
    end

    hold(data.ax, 'off');
    guidata(fig, data); % Save changes to 'data'
end

function fig = getFigureHandleByName(figName)
    figs = findall(groot, 'Type', 'figure');
    fig = [];
    for i = 1:length(figs)
        if strcmp(figs(i).Name, figName)
            fig = figs(i);
            break;
        end
    end
    if isempty(fig)
        error('Figure with name "%s" not found.', figName);
    end
end

function test_classification(~, ~)
    fig = getFigureHandleByName('Slice Viewer'); % Get the current figure handle
    data = guidata(fig); % Retrieve the data structure
    gmList = data.pointsList_gm;
    vesList = data.pointsList_ves;
    [labeled_array, ~] = classify_3D(data.dataArray, vesList, gmList, 0, 'ModelType', data.classification_model);    
    labeled_array = logical(labeled_array);
    
    %filter small objects
    %minSize = 100000;
    %labeled_array = bwareaopen(labeled_array, minSize);

    % Show Volume
    labeled_array = labeled_array(end:-1:1, :, :); % correct orientation to match slice viewer
    volumeViewer(labeled_array); %display small labelled chunk
    assignin('base','labeled_array', labeled_array);
end

function full_classification_choice(~, ~)
    % Confirmation dialog
    choice = questdlg('This operation may take a long time. Are you sure you want to proceed?', ...
        'Confirm FULL 3D Volume Segmentation', ...
        'Yes', 'No', 'No');
    
    % Handle response
    switch choice
        case 'Yes'
            full_classification();
        case 'No'
            disp('FULL 3D Volume Segmentation cancelled.');
        otherwise
            disp('FULL 3D Volume Segmentation cancelled.');
    end
end

function full_classification(~, ~)
    fig = getFigureHandleByName('Slice Viewer'); % Get the current figure handle
    data = guidata(fig); % Retrieve the data structure
    gmList = data.pointsList_gm;
    vesList = data.pointsList_ves;
    
    % Parameters for chunking
    overlap = data.overlap;  % Overlap size to avoid edge artifacts, choose a suitable value
    X = data.X;
    Y = data.Y;
    Z = data.Z;
    
    % Pad the array on all sides by 'overlap' so border chunks also have extra size
    paddedArray = padarray(data.fullarray, [overlap, overlap, overlap], 0, 'both');
    
    % Calculate number of chunks along each dimension
    numChunksX = ceil(X / data.chunkSize);
    numChunksY = ceil(Y / data.chunkSize);
    numChunksZ = ceil(Z / data.chunkSize);

    % Total number of chunks
    totalChunks = numChunksX * numChunksY * numChunksZ;
    
    % Initialize full labeled array
    whole_labeled_array = logical(paddedArray);
    [~, model] = classify_3D(data.dataArray, vesList, gmList, 0, 'ModelType', data.classification_model);
    save('model_temp', 'model'); %save final model in-case this function fails for whatever reason
    
    disp(['Segmenting started at ', datestr(now, 'yyyy-mm-dd HH:MM:SS')]);
    % Iterate through chunks with overlap
    currentChunkIndex = 1; % Initialize current chunk index
    for i = 0:numChunksX-1
        for j = 0:numChunksY-1
            for k = 0:numChunksZ-1
                tic; % Start timer
                disp(['Processing Chunk ', num2str(currentChunkIndex), ' of ', num2str(totalChunks)]);
                % Calculate extended chunk indices, handling borders
                xStart = i*data.chunkSize + 1;
                yStart = j*data.chunkSize + 1;
                zStart = k*data.chunkSize + 1;

                % Calculate end indices, ensuring they do not exceed the size of paddedArray
                xEnd = min((i+1)*data.chunkSize + 2*overlap, size(paddedArray, 1));
                yEnd = min((j+1)*data.chunkSize + 2*overlap, size(paddedArray, 2));
                zEnd = min((k+1)*data.chunkSize + 2*overlap, size(paddedArray, 3));
    
                % Extract extended chunk
                currentChunk = paddedArray(xStart:xEnd, yStart:yEnd, zStart:zEnd);
    
                % Process chunk with classify_3D
                [labeledChunk, ~] = classify_3D(currentChunk, gmList, vesList, 1, 'Model', model, 'ModelType', data.classification_model);

    
                % Calculate the indices for the non-overlapping central part of the chunk
                centralXStart = overlap + 1;
                centralYStart = overlap + 1;
                centralZStart = overlap + 1;
                centralXEnd = xEnd - xStart + 1 - overlap;
                centralYEnd = yEnd - yStart + 1 - overlap;
                centralZEnd = zEnd - zStart + 1 - overlap;
    
                % Discard the overlapped borders and replace only the central part in the whole_labeled_array
                whole_labeled_array(xStart+overlap:xEnd-overlap, yStart+overlap:yEnd-overlap, zStart+overlap:zEnd-overlap) = ...
                    labeledChunk(centralXStart:centralXEnd, centralYStart:centralYEnd, centralZStart:centralZEnd);
                
                elapsedTime = toc; % Stop timer and get elapsed time
                disp(['Chunk done in ', num2str(elapsedTime), ' seconds.']);
                
                currentChunkIndex = currentChunkIndex + 1; % Update chunk index
            end
        end
    end
    % Remove the padding from the whole_labeled_array to get back to the original size
    whole_labeled_array = whole_labeled_array(overlap+1:end-overlap, overlap+1:end-overlap, overlap+1:end-overlap);
    
    % Trim the whole_labeled_array to its original size if necessary
    whole_labeled_array = whole_labeled_array(1:X, 1:Y, 1:Z);

    %Post-processing: Visualization or further analysis of whole_labeled_array
    % Create a subfolder named after the input .mat file (excluding the extension)
    subFolderName = fullfile(data.matPathName, erase(data.matFileName, '.mat'));
    if ~exist(subFolderName, 'dir')
        mkdir(subFolderName); % Create the directory if it doesn't exist
    end
    
    matfilename = strcat('labeled_', data.matFileName);
    % Construct the path for the new .mat file
    newMatFilePath = fullfile(subFolderName, matfilename);
    
    % Save the final volume in the subfolder
    whole_labeled_array = logical(whole_labeled_array);
    save(newMatFilePath, 'whole_labeled_array', 'model', '-v7.3');
    disp(['Saved final volume to ', newMatFilePath]);
            
    volumeViewer(whole_labeled_array); %display the finished array
    %connected_comp(whole_labeled_array, newMatFilePath); %offer the user the chance to remove small bits
end

% Callback function to clear points
function clearPoints(~, ~)
    fig = getFigureHandleByName('Slice Viewer');
    data = guidata(fig);

    % Filter points for the current slice and plane
    % Clear points for the current slice and plane
    switch data.selectedPlane
        case 2 % 'yz' plane, x is the slicing dimension
        % Identify rows where the first column equals data.sliceNumber
        rowsToDeletegm = data.pointsList_gm(:, 1) == data.sliceNumber;
        % Delete those rows from data.pointsList_gm
        data.pointsList_gm(rowsToDeletegm, :) = [];

        % Identify rows where the first column equals data.sliceNumber
        rowsToDeleteves = data.pointsList_ves(:, 1) == data.sliceNumber;
        % Delete those rows from data.pointsList_gm
        data.pointsList_ves(rowsToDeleteves, :) = [];
        case 1 % 'xz' plane, y is the slicing dimension
        % Identify rows where the first column equals data.sliceNumber
        rowsToDeletegm = data.pointsList_gm(:, 2) == data.sliceNumber;
        % Delete those rows from data.pointsList_gm
        data.pointsList_gm(rowsToDeletegm, :) = [];

        % Identify rows where the first column equals data.sliceNumber
        rowsToDeleteves = data.pointsList_ves(:, 2) == data.sliceNumber;
        % Delete those rows from data.pointsList_gm
        data.pointsList_ves(rowsToDeleteves, :) = [];
        case 3 % 'xy' plane, z is the slicing dimension
        % Identify rows where the first column equals data.sliceNumber
        rowsToDeletegm = data.pointsList_gm(:, 3) == data.sliceNumber;
        % Delete those rows from data.pointsList_gm
        data.pointsList_gm(rowsToDeletegm, :) = [];

        % Identify rows where the first column equals data.sliceNumber
        rowsToDeleteves = data.pointsList_ves(:, 3) == data.sliceNumber;
        % Delete those rows from data.pointsList_gm
        data.pointsList_ves(rowsToDeleteves, :) = [];
    end
    data.pointsList_ves;
    guidata(fig, data); % Save changes back to the figure
    displaySlice(data.sliceNumber, data, fig); % Refresh the display
end