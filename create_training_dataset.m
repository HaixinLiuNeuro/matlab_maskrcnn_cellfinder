% script to creat a training dataset for mask r cnn training in matlab
clear;
root_fn = 'C:\HL\Github\'; % root folder for the project
image_dataset_fn = 'matlab_maskrcnn\imageset'; % location for the training dataset

im_fd = fullfile(root_fn, image_dataset_fn);
image_fns = dir(fullfile(im_fd, 'syth_image*.tiff'));
label_fns = dir(fullfile(im_fd, 'syth_label*.tiff'));

%% process label_fns to generate data structure for training 

%% define some parameters: 
%{
imageSize in meta file creation step
%}
imageSize = [512 512 3]; % to scale up to try
%% create the meta files: loop
out_fns = [];
for i_f = 1:length(image_fns)
    image_fn = fullfile(im_fd, image_fns(i_f).name);
    label_fn = fullfile(im_fd, label_fns(i_f).name);
    out_fn = fullfile(im_fd, [image_fns(i_f).name(1:strfind(image_fns(i_f).name, '.tiff')) 'mat'] );
    syth_image_make_meta_file(image_fn, label_fn, out_fn)
    out_fns{i_f} = out_fn; 

    if i_f ==1 
        tmp_im = imread(image_fn);
        % imageSize = [size(tmp_im) 3];
    end

end
%% imageDatastore
ds = fileDatastore(out_fns, 'ReadFcn',@(x)syth_image_reader2ds(x, imageSize));
%%
disp('Data preview')
data = preview(ds);
disp(data);
disp('Label')
disp(data{3}')

