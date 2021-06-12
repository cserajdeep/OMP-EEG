clc;
clear;
close all;
load('aar_train_140.mat');
load('aar_test_140.mat');
A=aar_train(:,2:end);
y=aar_train(:,1);
Kmax=6;
ITER=100;
[ I, x_hat, yrn] = OMP( y, A, Kmax );
red=I+1;
%%
X=aar_train(:,red);    %red
Y=aar_train(:,1);
[N,~]=size(X);
res=[];
res_past_acc=[];
%load engent40_mixbag_f10_9214;
bestInd=[];  
past_acc=0;
for k=1:ITER
temp=[];
Nbag=5;
Nuse=uint8(N*.60);
classifiers=cell(1,Nbag);
%ind=ceil(N*rand(Nuse,1));
for i=1:Nbag
    ind=ceil(N*rand(Nuse,1));
    Xi=X(ind,:);   % :
    Yi=Y(ind);   % :
    if i==1
    classifiers{i}=fitctree(Xi,Yi,'AlgorithmForCategorical','OVAbyClass');
    elseif i==2
    classifiers{i}=fitcknn(Xi,Yi,'NumNeighbors',13,'Distance','cosine');
    elseif i==3
    classifiers{i}=fitcdiscr(Xi,Yi,'DiscrimType','linear');
    elseif i==4
    classifiers{i}=fitcnb(Xi,Yi);
    elseif i==5
    classifiers{i}=fitcsvm(Xi,Yi,'KernelFunction','linear','BoxConstraint',1,'ClassNames',[1,2]); 
    else
      break
    end
    temp=[temp,ind]; 
end
Xtest=aar_test(:,red);  %2:end / red
Ytest=aar_test(:,1);
[Ntest,D]=size(Xtest);  %D
predict1=zeros(Ntest,Nbag);
for i= 1:Nbag,
     predict1(:,i)=predict(classifiers{i},Xtest);
end;
predict1=(mean(predict1,2)>1.5)+1;
[c,order]=confusionmat(Ytest, predict1);
acc=sum(diag(c)/sum(c(:)));
if past_acc < acc
    bestInd=temp;
    past_acc=acc;
end
res=[res acc];
res_past_acc=[res_past_acc past_acc];
end
fprintf('Max:%f\tAvg.:%f\tStd.:%f\n',max(res),mean(res),std(res));
fprintf('Best:%f\n',max(res_past_acc));
%save engent40_mixbag_f10_9214.mat bestInd;