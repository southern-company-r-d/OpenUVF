function manual_classification()
% UVF_ManualClassification - Tool to classify UVF Image Sets
%
% Braden Gilleland - Southern Company Services R&D 2018
%
% UVF_ManualClassification provides a simple interface to manually classify
% cells and assign them to specific folders for further review and
% pre-training labeling. The tool should simplify the workflow of labeling
% images. The tool goes module by module, allowing the classification of
% the modules and its cells into their respective output folders for
% further review in the course of developing the
% training/validation/analysis set

%% Function Setup

    %Input Directory Setup
    inputDirectoryMain = 'S:\Workgroups\SCS Research and Development\Research\BHGilleland\Projects\UV Fluorescence\Processing Suite\Image Segmentation\outputs\';
    inputDirectoryCellBinary = 'Cells - Binary/';
    inputDirectoryCellColor = 'Cells - Color/';
    inputDirectoryModBinary = 'Modules - Binary/';
    inputDirectoryModColor = 'Modules - Color/';
    inputDirectoryModData = 'Modules - Data/';
    
    
    %Output Directory Setup
    outputDirectoryMain = 'S:\Workgroups\SCS Research and Development\Research\BHGilleland\Projects\UV Fluorescence\Processing Suite\Images\';
    outputDirectoryCellBinaryCracked = 'Cells - Binary/Cracked/';
    outputDirectoryCellBinaryNotCracked = 'Cells - Binary/Not Cracked/';
    outputDirectoryCellColorCracked = 'Cells - Color/Cracked/';
    outputDirectoryCellColorNotCracked = 'Cells - Color/Not Cracked/';
    outputDirectoryModBinary = 'Modules - Binary/';
    outputDirectoryModColor = 'Modules - Color/';
    outputDirectoryModData = 'Modules - Data/';
    
    %Image Settings
    inputFileType = '.PNG';
    nCells = 72;
           
    %Review Preferences
    startInd = 343;


%% Function Initialization

    %Loading directories
    modules.binarydir = dir([inputDirectoryMain, inputDirectoryModBinary,'*',inputFileType]);
    modules.colordir = dir([inputDirectoryMain, inputDirectoryModColor,'*',inputFileType]);
    modules.datadir = dir([inputDirectoryMain, inputDirectoryModData,'*',inputFileType]);
    cells.binarydir = dir([inputDirectoryMain, inputDirectoryCellBinary,'*',inputFileType]);
    cells.colordir = dir([inputDirectoryMain, inputDirectoryCellColor,'*',inputFileType]);

    %Figure Initialization
    figure('Name', 'UVF Manual Classification', 'Units', 'Normalized', 'OuterPosition', [ 0 0 1 1]);
    
    
%% Processing

    for m = startInd:length(modules.binarydir)
        
        %Identifying image to load
        imname = modules.binarydir(m).name;
        modules.name = strsplit(imname, '.');
        modules.name = modules.name{1};
        
        %Load and display Module Color Image
        modules.colorim = imread(fullfile(modules.colordir(m).folder, imname));
        subplot(2,3,[1 4]);
        imshow(modules.colorim);
        axis equal
        title('Module Color');       
        
        %Load and display Module Binary Image
        modules.binaryim = imread(fullfile(modules.binarydir(m).folder, imname));
        subplot(2,3, [2 5]);
        imshow(modules.binaryim);
        axis equal
        title('Module Binary');
        
        %Ask user if module should be considered
        
        prompt = sprintf('Module %s Displayed. Good to continue to cells? (y/n)    ', modules.name);
        response = input(prompt, 's');
        
        %Iterating through cells for classification
        if strcmpi(response, 'y')
            
            %Save Module Images
            modules.binaryoutfn = [outputDirectoryMain, outputDirectoryModBinary, imname];
            modules.coloroutfn = [outputDirectoryMain, outputDirectoryModColor, imname];
            imwrite(modules.binaryim, modules.binaryoutfn);
            imwrite(modules.colorim, modules.coloroutfn);      
            
            %Iterate through module cells
            for c = 1:72
                validInput = false;
                while ~validInput
                    %Identify cell name
                    cell.name = [modules.name,'_C', num2str(c)];

                    %Load and display Color Image
                    cell.colorim = imread(fullfile(cells.colordir(m).folder, [cell.name, inputFileType]));
                    subplot(2,3,3);
                    imshow(cell.colorim);
                    axis square
                    title('Cell Color');

                    %Load and display Binary Image
                    cell.binaryim = imread(fullfile(cells.binarydir(m).folder, [cell.name, inputFileType]));
                    subplot(2,3,6);
                    imshow(cell.binaryim);
                    axis square
                    title('Cell Binary');

                    %Ask user what to do with cell
                    prompt = 'Not Cracked, Cracked or Toss? (1,2,3)    ';
                    response = input(prompt, 's');

                    %Handling cell
                    switch response
                        case '1'
                            cell.colorfn = [outputDirectoryMain, outputDirectoryCellColorNotCracked, cell.name, inputFileType];
                            cell.binaryfn = [outputDirectoryMain, outputDirectoryCellBinaryNotCracked, cell.name, inputFileType];
                            imwrite(cell.colorim, cell.colorfn);
                            imwrite(cell.binaryim, cell.binaryfn);                        
                            validInput = true;
                        case '2'
                            cell.colorfn = [outputDirectoryMain, outputDirectoryCellColorCracked, cell.name, inputFileType];
                            cell.binaryfn = [outputDirectoryMain, outputDirectoryCellBinaryCracked, cell.name, inputFileType];
                            imwrite(cell.colorim, cell.colorfn);
                            imwrite(cell.binaryim, cell.binaryfn);
                            validInput = true;
                        case '3'                        
                            validInput = true;
                    end
                end
                
            end
        end



    end








end