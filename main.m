function [  ] = main( imname )


% Close all figures.
 delete(findall(0,'Type','figure'))
 bdclose('all')

warning off;

%% Step 1
% Loading
I=imread(imname);
figure(1);
imshow(I);
title('Input Image');


%% Step 2
% Preprocessing
figure(2);
I2=rgb2gray(I);
subplot(1,3,1);
imshow(I2);
title('Grayscaled Image');

I3=medfilt2(I2,[3 3]);
subplot(1,3,2);
imshow(I3);
title('Filtered Image');

I31=histeq(I3);
subplot(1,3,3);
imshow(I31);
title('Histogramed Image');


%% Step 3
% Seperating Out the Infected Cells(Blue Plane)

figure(3);
bluecells = I(:,:,3)  - 0.5*(I(:,:,1)) - 0.5*(I(:,:,2));
subplot(1,2,1);
[m,n] = size(bluecells);

temp = zeros(m,n);
for i=1:m
    for j=1:n
      if (bluecells(i,j) > 0)
        temp(i,j) = 1;
      end
    end
end

imshow(bluecells);
title('Seperation');

%  BW = temp;
% subplot(2,2,3);
% imshow(BW);
% title('EXTRACTION-0');


Blue = bluecells > 10;
subplot(1,2,2);
imshow(Blue);
title('RBC and Infected Cells');


%% Step 4
% Noise Removal

figure(4);
NRem = bwareaopen(Blue, 10);
subplot(2,2,1)
imshow(NRem);
title('Noise Removal');

I5=imadjust(I3);
subplot(2,2,2);
imshow(I5);
title('Intensity Adjustment');

f=graythresh(I2);

I6=im2bw(I5,f);
subplot(2,2,3);
imshow(I6);
title('Binary Image');

%% Step 5
% Morphology

figure(5);
I7=bwareaopen(I6,20);
subplot(3,3,1);
imshow(I7);
title('Segmentation');

[~, thd] = edge(I2, 'sobel');
ha = 0.5;
seg = edge(I2,'sobel', thd * ha);
subplot(3,3,2);
imshow(seg);
title('Gradient');

linepp = strel('line', 3, 90);
linepl = strel('line', 3, 0);

segdil = imdilate(seg, [linepp linepl]);
subplot(3,3,3);
imshow(segdil);
title('Dilated image');

segf = imfill(segdil, 'holes');
subplot(3,3,4);
imshow(segf);
title('Holes Filled');


I2=rgb2gray(I);
Iadeq = adapthisteq(I2);
subplot(3,3,5);
imshow(Iadeq);
title('Adaptive Histogramed Image');

ad = im2bw(Iadeq, graythresh(Iadeq));
adp = imfill(ad,'holes');
adpt = imopen(adp, ones(5,5));

adptv = bwareaopen(adpt, 5);
adper = bwperim(adptv);
comb = imoverlay(Iadeq, adper, [.3 1 .3]);
subplot(3,3,6);
imshow(comb);
title('Combined Image');

mimg = imextendedmax(Iadeq, 80);
subplot(3,3,7);
imshow(mimg);
title('Masked Image');

%%title('MASKED IMAGE');
mimg = imclose(mimg, ones(5,5));
% subplot(3,3,8);imshow(mask_em);
% title('mask em');

% mask_em1 = imfill(mask_em, 'holes');
% subplot(3,3,8);imshow(mask_em1);
% title('mask em1');
% mask_em = bwareaopen(mask_em, 5);
% subplot(3,3,8);imshow(mask_em);
% title('mask em');


%% Step 6
% RBC Count

mimg5 = imcomplement(mimg);
mimg5 = bwareaopen(mimg5,2500);

I9=bwlabel(mimg5);
%I9 = bwareaopen(I9,500);

rbc=max(max((I9)));
RBC = rbc

figure(6);
imshow(I9);
title('Total Number of Cells')

I8=imfill(I7,'holes');

L=bwlabel(I8);
%% Step 7
% Infected Cells Count

infcells=max(max(L));
Infected_Cells = infcells

Ratio = infcells/rbc


% The End

end
