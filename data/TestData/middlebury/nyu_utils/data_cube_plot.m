scatter3(64*rand(1000,1),256*rand(1000,1),64*rand(1000,1),'.')
xlim([0 64]); zlim([0 64]);
ylim([0 256]);

[X,Y] = meshgrid(1:64,1:64);
X = X(:); Y = Y(:);
mask = rand(size(X));
mask(mask < 0.8) = 0;
mask = logical(mask);
Z = 128*ones(size(X(:)));
hold on; scatter3(X(mask),Z(mask),Y(mask),'.');
hold off;
camup([0 1 0]);
set(gca,'xtick',[],'ytick',[],'ztick',[]);
box on;
set(gcf,'color','white');
axis image;
export_fig('datacube2.pdf');