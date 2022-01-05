%addpath('/share/apps/toast-2.0.2/linux64/mex2/')
%addpath(genpath('/home/gdisciac/toastpp/script/matlab/'))

toastdir =  '/home/gdisciac/toastpp/'
toastver = '/share/apps/toast-2.0.2/linux64'
nogui = 0;

% Add all directories under the script/matlab node
p = genpath([toastdir '/script/matlab/']);
p(find(p==' ')) = '^';     % protect spaces in directory names
p(find(p==pathsep)) = ' '; % replace separators with spaces
p = textscan(p,'%s');      % split path into separate elements
p = p{1};
for i = 1:size(p,1)        % restore spaces
    p{i}(find(p{i}=='^')) = ' ';
end

k1=strfind(p,'CVS');        % eliminate CVS subdirs
k2=strfind(p,'.svn');       % eliminate .svn subdirs
k = [k1,k2];

pth = '';
for i=1:size(k,1)
    if length(k{i,1}) == 0 && length(k{i,2}) == 0
        if length(pth) > 0
            pth = [pth pathsep];
        end
        pth = [pth cell2mat(p(i))];
        if ~nogui
            disp(['Adding search path ' cell2mat(p(i))])
        end
    end
end
addpath (pth);

% add the mex directory
mexp = [toastver '/mex2'];
addpath (mexp);
if ~nogui
    disp(['Adding search path ' mexp])
end

rmpath ([toastdir '/script/matlab/toast']); % remove original toast script directory

cd('/home/gdisciac/SOLUS/example/VICTRE_PARADIGM');


