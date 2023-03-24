function syth_image_make_meta_file(image_fn, label_fn, out_fn)
% reserve for saving into .m files
data = [];
%
% label_fn = fullfile(im_fd, label_fn(1).name)
tmp_im_label = imread(label_fn);

% imshow(tmp_im_label, [0 max(tmp_im_label(:))])

%% image file name (fullpath)
data.image_fn = image_fn;
%% bboxes:
%{
  big assumption here, non-overlapping cells (processed during image generation process)
  in the future need to improve/ or just write meta file from image
  generation process
  Thus, can just use index coding to get ROI mask
%}
%% labels: default as a cell ROI
%% masks: use coded ROI number
%{

%% reverse engineer
% [B,L] = bwboundaries(tmp_im_label > 0);

%{
figure; imshow(BW); hold on; 
% Loop through object boundaries  
for k = 1:N 
    % Boundary k is the parent of a hole if the k-th column 
    % of the adjacency matrix A contains a non-zero element 
    if (nnz(A(:,k)) > 0) 
        boundary = B{k}; 
        plot(boundary(:,2),... 
            boundary(:,1),'r','LineWidth',2); 
        % Loop through the children of boundary k 
        for l = find(A(:,k))' 
            boundary = B{l}; 
            plot(boundary(:,2),... 
                boundary(:,1),'g','LineWidth',2); 
        end 
    end 
end
%}
%}
n_roi = max(tmp_im_label(:));
data.masks = zeros(size(tmp_im_label,1), size(tmp_im_label,2), n_roi, 'logical');
data.bbox = nan(n_roi, 4);
for i_cell = 1:n_roi
    data.masks(:,:, i_cell) = tmp_im_label == i_cell;
    % to make sure the bbox correspond to each ROI/cell
    st = regionprops(data.masks(:,:, i_cell) > 0, 'BoundingBox' );
    data.bbox(i_cell,:) = st.BoundingBox;
end
data.bbox = round(data.bbox);
data.label = categorical(repmat({'cell'},n_roi,1));

%% write to file
save(out_fn, '-struct', 'data');