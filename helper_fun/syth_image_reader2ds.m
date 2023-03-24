function  out = syth_image_reader2ds(fn, targetSize)
data = load(fn);
% read in the image
im = imread(data.image_fn);
%% dealing with resize (here define prop, no crop needed)
if nargin > 1
    imgSize = size(im);
    masks = data.masks;
    % bbox = data.bbox;
    % Resize images, masks and bboxes
    [~, minDim] = min(imgSize(1:2));

    resizeSize = [NaN NaN];
    resizeSize(minDim) = targetSize(minDim);

    im = imresize(im,resizeSize);
    data.masks = imresize(masks,resizeSize);
    % redo bbox
    for i_cell = 1:size(data.masks,3)
        
        % to make sure the bbox correspond to each ROI/cell
        st = regionprops(data.masks(:,:, i_cell) > 0, 'BoundingBox' );
        data.bbox(i_cell,:) = st.BoundingBox;
    end
    

end
%% OUTPUT
% For grayscale images repeat the image across the RGB channel
if(size(im,3)==1)
    im = repmat(im, [1 1 3]);
end
out{1} = im;
out{2} = data.bbox;
out{3} = data.label;
out{4} = data.masks;

