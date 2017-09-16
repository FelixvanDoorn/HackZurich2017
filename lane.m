clear all
im = imread('testimages/im5.jpg');


imgrey = rgb2gray(im);


imcrop = imgrey(400:end, 1:end);
imshow(imcrop);

bordersize = 4
img = uint8(ones(size(imcrop)+[bordersize*2, bordersize*2])*10);
img(bordersize+1:end-bordersize, bordersize+1:end-bordersize) = imcrop; 


imshow(img);



imedge = edge(img,'Canny',[0.08, 0.3], 3);


%corner hack
size = bordersize * 8;
imedge(1:size, 1:size) = 0;
imedge(end-size:end, 1:size) = 0;
imedge(1:size, end-size:end) = 0;
imedge(end-size:end, end-size:end) = 0;




se = strel('disk',4);
imedge = imdilate(imedge,se);
figure, imshow(imedge)


imedge = imfill(imedge, 'holes');
se = strel('disk',10);
imedge = imopen(imedge,se);


figure,imshow(imedge)

imsmooth = imgaussfilt(uint8(imedge)*255,20);
figure,imshow(imsmooth)

%s = regionprops((img_filled), 'centroid','eccentricity','Area', 'FilledImage', 'FilledArea','Perimeter', 'Solidity', 'PixelIdxList');


