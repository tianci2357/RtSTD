%% HAD
close all;clear;clc;

%%
addpath('./DATASET');
addpath('./data2');

addpath('./CRD');
addpath(genpath('./KIFD'));
addpath('./LRASR');
addpath('./LRX');
addpath(genpath('LSC-TV'));
addpath('./LSMAD'); 

addpath('./iForest'); % in SSII
addpath(genpath('./KPCA')); % in SSII
addpath('./PTA_LiWei'); % in SSII
addpath('./Gabor'); % in SSII
addpath('./3DROC');  % in SSII
addpath('./IIFD');  % in SSII

%% load data and instation
% data2  68cla05.tif 68clx05.tif 681056002.tif 681101003.tif
% 1031005.tif 1032005.tif  681058001.tif 681059005.tif 681101001.tif
filename = '68cla05.tif';   % SanDiegoI   aviris150   Hyperion    Pavia                    
load (filename)

[Length,Width,Band]=size(data);

L=10;
Limit=2^L-1;

IMsz = Length * Width; % single band size 
data_mxn=reshape(data,IMsz,Band);

if exist('map','var')
else
    map=groundtruth;
    map_mx1=reshape(map,[],1);
end

if exist('groundtruth','var')
else
    groundtruth=map;
end

figure('name','data'),imshow(data(:,:,1),[ ]);
figure('name','map'),imshow(map,[]);

Methods = 0;

%%  LRX detector
Methods = Methods+1;
Mname='LRX';
fprintf('Methods %d: %s\n',Methods,Mname);
tic
LRX1 = LRXfc(data,3,5,1e4);
toc
figure('name','LRX');imshow(LRX1);colormap(jet);


%%  CRD detector
Methods = Methods+1;
Mname='CRD';
fprintf('Methods %d: %s\n',Methods,Mname);

tic
CRD2=CRDfc(data,3,7,1e5);
toc
figure('name','CRD'), imshow(CRD2);colormap(jet) ;


%%  LRASR detector
Methods = Methods+1;
Mname='LRASR';
fprintf('Methods %d: %s\n',Methods,Mname);
tic
[Length,Width,Band]=size(data);
datat=reshape(data,Length*Width,Band);

D=dictionary_func(datat,Band,1);  %p=20,D=0
data2=datat';
[S,E,J] = LADMAP_LRASR(data2,D,0.1,0.1);
anomaly=zeros(Length,Width);
E=E';
sz=[Length,Width,Band];
EE=reshape(E,sz);
for i=1:Length
    for j=1:Width
        anomaly(i,j)=norm(reshape(EE(i,j,:),1,Band),2);
    end   
end  
      
LRASR3 = (anomaly-min(anomaly(:)))/(max(anomaly(:))-min(anomaly(:)));       
toc
figure('name','LRASR');imshow(LRASR3);colormap(jet);


%%  LSMAD detector
Methods = Methods+1;
Mname='LSMAD';
fprintf('Methods %d: %s\n',Methods,Mname);
tic
[length,width,band]=size(data);
datat=reshape(data,length*width,band);
[B,S] = LSMADfunc(datat,1,round(0.5*length*width),0.2,30);
szt=[length,width,band];
SS=reshape(S,szt);
anomaly=zeros(length,width);
for i=1:length
    for j=1:width
        anomaly(i,j)=norm(reshape(SS(i,j,:),1,band),2);
    end   
end        
LSMAD4 = (anomaly-min(anomaly(:)))/(max(anomaly(:))-min(anomaly(:)));  
toc
figure('name','LSMAD');imshow(LSMAD4);colormap(jet);


%%  PTA with LTV-norm detector
Methods = Methods+1;
Mname='PTA';
fprintf('Methods %d: %s\n',Methods,Mname);
tic
%------------------------normalization-----------------------
DataTest =data;

mask = map;
mask2 = map_mx1;
[Length, Width, Band] = size(DataTest);  
for i=1:Band
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end
IMsz = Length * Width;
toc
tic
%-----------------------Start testing
mask_reshape = reshape(mask, Length*Width, 1);
anomaly_map = logical(double(mask_reshape)>0  );
normal_map = logical(double(mask_reshape)==0 );
DataRp = reshape(DataTest,IMsz,Band); 

%----------------------
tol1 = 1e-4;
tol2 = 1e-6;
maxiter = 400;
truncate_rank = 1;
alphia = 1.7;  % 1.7
beta = 0.069;   %0.069
tau =0.1;       %0.1
toc
tic
[X,S,area_PTA] = AD_Tensor_LILU1(DataTest,alphia,beta,tau,truncate_rank,maxiter,tol1,tol2,normal_map,anomaly_map);

PTA5 = sqrt(sum(S.^2,3));
PTA5 = (PTA5-min(PTA5(:)))/(max(PTA5(:))-min(PTA5(:)));
r_PTA = reshape(PTA5,IMsz,1);
toc
figure('name','PTA');imshow(PTA5,[]);colormap(jet) 

%%  IIFD detector
Methods = Methods+1;
Mname='IIFD';
fprintf('Methods %d: %s\n',Methods,Mname);
tic
[Data_PCA_3D] = pca(DataRp, 9);
Data_PCA_3D = Data_PCA_3D';
Data_PCA = reshape(Data_PCA_3D(:,1),Length, Width);
gaborArray = gaborFilterBank(5,8,199,199); % Generates Gabor filter bank; 
[ ~,featureVector] = gaborFeatures(Data_PCA,gaborArray,Length,Width);   % Extracts Gabor feature vector, 'featureVector', from the image,
Gabor_feature = reshape(featureVector,Length*Width,40) ;
toc
%-------------------------------------------------------------------------
NumTree = 32;  
K = 5;  

tic
[PMass_spectral,QMass_spectral,score_LIIF, ~, ~] = Local_IIFD(DataRp, DataRp, map_mx1, NumTree, Length, Width); % fig in
toc

tic
[PMass_spatial,QMass_spatial,score_GabIIF, AUC, ~] = IIFD_Gabor(Gabor_feature, map_mx1, NumTree, K);
toc

PMass_final = 0.618*PMass_spectral + 0.382*PMass_spatial;  
QMass_final = 0.618*QMass_spectral + 0.382*QMass_spatial;
ratio = PMass_final ./ QMass_final;
scoreraw = mean(ratio, 2);
value = 2 * (log(IMsz-1)+0.5772156649) - 2*(IMsz-1) / IMsz;  
score_SSIIFD = 1 - 2.^(-scoreraw./value);

LIIFD_show = reshape(score_LIIF, Length, Width);
figure('name','LIIFD');imshow(LIIFD_show,[]);colormap(jet) 

% [TPR_CRD,FPR_CRD] = roc(reshape(groundtruth,[],1),reshape(LIIFD_show,[],1));
% % b=sort(FPR_CRD');
% AUC_CRD = polyarea([0,sort(FPR_CRD','ascend'),1,1],[0,sort(TPR_CRD','ascend'),1,0]);
% sAUC = num2str(AUC_CRD);
% algo = 'LIIFD AUC:';
% tit = [algo sAUC];
% figure('name','LIIFD  ROC');semilogx(FPR_CRD,TPR_CRD,'-','LineWidth',3);title(tit);
% ylim([0,1]); 

IFD_spatial_show = reshape(score_GabIIF, Length, Width);
figure('name','IIF_Gabor');imshow(IFD_spatial_show,[]);colormap(jet) 

SSIIFD6 = reshape(score_SSIIFD, Length, Width);
figure('name','SSIIFD'); imshow(SSIIFD6,[]);colormap(jet) 


%%  CJJC
Methods = Methods+1;
Mname='CJJC';
fprintf('Methods %d: %s\n',Methods,Mname);

CJJC=result;
figure('name','CJJC'); imshow(CJJC,[]);colormap(jet) 

%%  detector
% Methods = Methods+1;
% Mname='CRD';
% fprintf('Methods %d: %s\n',Methods,Mname);
%% ROC AUC
% 颜色定义
[all_themes, all_colors] = GetColors();

fig = figure('name','ROC');

[TPR_CRD,FPR_CRD] = roc(reshape(groundtruth,Length*Width,1),reshape(LRX1,Length*Width,1));
AUC_LRX1 = polyarea([0,sort(FPR_CRD','ascend'),1,1],[0,sort(TPR_CRD','ascend'),1,0]);
semilogx(FPR_CRD,TPR_CRD,'-','Color',all_colors(1, :),'LineWidth',2);hold on;

[TPR_CRD,FPR_CRD] = roc(reshape(groundtruth,Length*Width,1),reshape(CRD2,Length*Width,1));
AUC_CRD2 = polyarea([0,sort(FPR_CRD','ascend'),1,1],[0,sort(TPR_CRD','ascend'),1,0]);
semilogx(FPR_CRD,TPR_CRD,'--','Color',all_colors(2, :),'LineWidth',2);hold on;

[TPR_CRD,FPR_CRD] = roc(reshape(groundtruth,Length*Width,1),reshape(LRASR3,Length*Width,1));
AUC_LRASR3 = polyarea([0,sort(FPR_CRD','ascend'),1,1],[0,sort(TPR_CRD','ascend'),1,0]);
semilogx(FPR_CRD,TPR_CRD,':','Color',all_colors(3, :),'LineWidth',2);hold on;

[TPR_CRD,FPR_CRD] = roc(reshape(groundtruth,[],1),reshape(LSMAD4,[],1));
AUC_LSMAD4 = polyarea([0,sort(FPR_CRD','ascend'),1,1],[0,sort(TPR_CRD','ascend'),1,0]);
semilogx(FPR_CRD,TPR_CRD,'-','Color',all_colors(4, :),'LineWidth',2);hold on;

[TPR_CRD,FPR_CRD] = roc(reshape(groundtruth,[],1),reshape(PTA5,[],1));;
AUC_PTA5 = polyarea([0,sort(FPR_CRD','ascend'),1,1],[0,sort(TPR_CRD','ascend'),1,0]);
semilogx(FPR_CRD,TPR_CRD,'--','Color',all_colors(5, :),'LineWidth',2);hold on;

[TPR_CRD,FPR_CRD] = roc(reshape(groundtruth,[],1),reshape(SSIIFD6,[],1));
AUC_SSIIFD6 = polyarea([0,sort(FPR_CRD','ascend'),1,1],[0,sort(TPR_CRD','ascend'),1,0]);
semilogx(FPR_CRD,TPR_CRD,':','Color',all_colors(6, :),'LineWidth',2);hold on;

[TPR_CRD,FPR_CRD] = roc(reshape(groundtruth,Length*Width,1),reshape(CJJC,Length*Width,1));
AUC_CJJC = polyarea([0,sort(FPR_CRD','ascend'),1,1],[0,sort(TPR_CRD','ascend'),1,0]);
semilogx(FPR_CRD,TPR_CRD,'-','Color',all_colors(7, :),'LineWidth',2);
title('Case 2');

ylim([0,1]);
xlim([9.9*10e-7,1]);
ylabel('Detection rate');
xlabel('False alarm rate');
legend({'LRX', 'CRD', 'LRASR', 'LSMAD', 'PTA', 'SSIIFD', 'Proposed'}); % 添加图例
%%
AUCall=[AUC_LRX1,AUC_CRD2,AUC_LRASR3,AUC_LSMAD4,AUC_PTA5,AUC_SSIIFD6,AUC_CJJC];

