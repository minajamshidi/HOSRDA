% April 2016
% By Mina Jamshidi Idaji
% M.Sc. Thesis
% EE Department,Sharif University of Technology
%--------------------------------------------------------------------------
% These Codes have been written in MATLAB R2014a (8.3.0.532), maci64
%--------------------------------------------------------------------------
% this function runs the algorithm of learning the HOSRDA subspace.

% Options:
%Options.NTr = number of training character used .
%Options.NN = number of repetitions for each stimulation in p300 speller.
%Options.J = number of features in HOSRDA.
%Options.maxiter = maximum number of iterations.
%Options.F_tol = tolorance of Error for stop criterion.
%Options.ReguAlpha = regularization parameter.
%%
function [U,feat_train,F] = HOSRDA(X_train,LBL_train,Options)
NTr = Options.NTr;
NN = Options.NN;
% J = Options.J;
ReguAlpha = Options.ReguAlpha;
maxiter = Options.maxiter;
F_tol = Options.F_tol;
JJ = Options.JJ;
% JJ = [J,J];



Ntrial_eachChar = 12*NN;%number of trials for each character.
X_train1 = X_train;
LBL_train1 = LBL_train;
X_train = X_train1(:,:,1:NTr*Ntrial_eachChar);
LBL_train = LBL_train1(1:NTr*Ntrial_eachChar);
clear X_train1 LBL_train1;
%%
tsize = size(X_train);
N = ndims(X_train);
K = tsize(N);
classlbl = unique(LBL_train);
classnum = length(classlbl);
%% training
fprintf('     ~Training HOSRDA Subspace Basis Factors...\n');


% Init------------------ ------------------ ------------------ ------------
tic
U=cell(1,N-1);
X_train_tensor = tensor(X_train);
for n = 1:N-1
    %         U{n} = randn(tsize(n),R(n));
    U{n} = nvecs(X_train_tensor,n,JJ(n));
end

X_barbar = mean(X_train,N);
X_c_barbar = cell(1,classnum);
mc = zeros(classnum,1);
for c = 1:classnum
    idx = find(LBL_train==classlbl(c));
    mc(c) = length(idx);
    X_idx = X_train(:,:,idx)-repmat(X_barbar,1,1,mc(c));
    X_c_barbar{c} = X_idx;
end
fprintf('first stage %g s\n',toc)
F = zeros(maxiter+1,1);
F(1) = FisherRatio(X_train_tensor,U,LBL_train,N,K);
% HOSRDA------------------ ------------------ ------------------ ----------
tic
for iter = 1:maxiter
    
    for n = 1:N-1
        dims = 1:N-1; dims(n) = [];
        Pn = prod(JJ(dims));
        H = [];
        y = zeros(K*Pn,JJ(n));
        for c = 1:classnum
            Xc = tensor(cell2mat(X_c_barbar(c)));
            
            Hc = ttm(Xc,U(dims),dims,'t');
            H = cat(N,H,double(Hc));
            %             rand('state',0);
            v = rand(Pn,JJ(n));
            
            if c==1
                y( 1:mc(1)*Pn , :) = repmat(v,mc(c),1);
            else
                n1 = sum(mc(1:c-1))*Pn;
                y( n1+1:n1+mc(c)*Pn , :) = repmat(v,mc(c),1);
            end 
        end %for c
        %        y = [y,ones(size(y,1),1)];
        %        [y,~] = qr(y,0);
        %        y(:,1) = [];
        Hn = double(tenmat(tensor(H),n));
%         ddata = full(Hn*Hn');
% %         ddata = mtimesx(Hn,Hn');
%         
%         if ReguAlpha > 0
%             for j = 1:size(ddata,1)
%                 ddata(j,j) = ddata(j,j) + ReguAlpha;
%             end
%         end
%         B = Hn*y;  %*
%         R = chol(ddata);%*
%         Un = R\(R'\B);%*
%         [Un] = lsqr2( Hn', y, 0, 20,1e-5,1e-5);
        Un = Hn'\y;

        [U{n},~] = gs_m(Un);
        %
        %         ddata = Hn*Hn';
        %         B = Hn*y;
        %         R = chol(ddata);
        %         Un = R\(R'\B);
        %         [U{n},~] = gs_m(Un);
    end % for n
    Find = iter+1;
    F(Find) = FisherRatio(X_train_tensor,U,LBL_train,N,K);
    if abs(F(Find)-F(Find-1))<F_tol
        break
    end
end % for iter
fprintf('        ^Training HOSRDA Subspace took %g (s) and terminated after %g iterations\n',...
    toc,iter);

%% Train Data Projection
Feat_train = ttm(tensor(X_train),U,1:2,'t'); % Project tensor
Feat_train = Feat_train.data;
feat_train = reshape(Feat_train,[],size(Feat_train,ndims(Feat_train)))';

end %function