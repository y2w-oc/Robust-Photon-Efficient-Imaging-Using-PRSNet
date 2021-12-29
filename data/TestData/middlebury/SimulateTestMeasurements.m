clear all; close all; clc;
LOW_RES = 0;  % 0:HR 1:LR
addpath('./nyu_utils');
if LOW_RES
    outdir = './processed/LR';
else
    outdir = './processed/HR';
end
if ~exist(outdir, 'dir')
    mkdir(outdir)
end

scenedir = './raw/';
scenes = { 'Reindeer', 'Art', 'Plastic', 'Moebius', 'Laundry', ...
           'Dolls', 'Bowling1', 'Books' };
c = 3e8;
bin_size = 80e-12;
num_bins = 1024; 
res = 64;

% this is the 9 typical noise levels
simulation_params_T = [10 2;
                     5 2;                     
                     2 2;
                     10 10;
                     5 10;
                     2 10;
                     10 50;
                     5 50;
                     2 50];
        
% this is the extra 3 low SBR noise levels
simulation_params_E = [3 100;
                       2 100;
                       1 100];

t_s = tic;
for ss = 1:length(scenes)
    % for test, use 9typical or 3 extra low SBR noise levels
    for zz = 1:size(simulation_params_T,1) 
        fprintf('Processing scene %s...\n',scenes{ss});

        fid = fopen([scenedir scenes{ss} '/dmin.txt']);
        dmin = textscan(fid,'%f');
        dmin = dmin{1};
        fclose(fid);

        f = 3740;
        b = .1600;
        disparity = (single(imread([scenedir scenes{ss} '/disp1.png'])) + dmin);
        depth = f*b ./ disparity;    
        intensity = rgb2gray(im2double(imread([scenedir scenes{ss} '/view1.png'])));

        mask = disparity == dmin;
        depth(mask) = nan;

        if ss == 1
            %there is big patch whose intensity approximates zero in the Reindeer scene 
            mask = intensity < 0.003;
            intensity(mask) = 0.01;

        elseif ss == 7
            mask = disparity == dmin;
            se = strel('disk',3);
            mask = imerode(mask,se);
            se = strel('disk',20);
            mask = imdilate(mask,se) & disparity == dmin;
            imagesc(mask);
            depth(mask) = 1.962;            
        end

        depth = full(inpaint_nans(double(depth),5));
        imagesc(depth);
        drawnow;

        r1 = 64 - mod(size(depth,1),64);
        r2 = 64 - mod(size(depth,2),64);
        r1_l = floor(r1/2);
        r1_r = ceil(r1/2);
        r2_l = floor(r2/2);
        r2_r = ceil(r2/2);

        depth = padarray(depth, [r1_l r2_l], 'replicate', 'pre');
        depth = padarray(depth, [r1_r r2_r], 'replicate', 'post');
        intensity = padarray(intensity, [r1_l r2_l], 'replicate', 'pre');
        intensity = padarray(intensity, [r1_r r2_r], 'replicate', 'post');

        depth_hr = depth;
        intensity_hr = intensity;
        
        if LOW_RES
            depth = imresize(depth, 1/8, 'bicubic');
            intensity = imresize(intensity, 1/8, 'bicubic');
        end
        depth = max(depth, 0);
        intensity = max(intensity, 0);

        albedo = intensity;

        [x,y] = meshgrid(1:size(intensity,2),1:size(intensity,1));
        x = x - size(intensity,1)/2; 
        y = y - size(intensity,2)/2;
        X = x.*depth ./ f;
        Y = y.*depth ./ f;

        dist = sqrt(X.^2 + Y.^2 + depth.^2);
        clear x1 x2 y1 y2 X1 X2 Y1 Y2;

        % convert to time of flight
        d = depth;
        tof = depth * 2 / c;

        bins1 = size(depth,1);
        bins2 = size(depth,2);

        % convert to bin number
        range_bins = round(tof ./ bin_size);
        if any(reshape(range_bins > num_bins, 1, []))
            fprintf('some photon events out of range\n');
        end
        
        range_bins = min(range_bins, num_bins);   % gating the outliers
        range_bins = max(range_bins, 1);  % the GT depth in 2D

        mean_signal_photons = simulation_params_T(zz,1);
        mean_background_photons = simulation_params_T(zz,2);
        SBR = mean_signal_photons ./ mean_background_photons;
        disp(['The mean_signal_photons: ', num2str(mean_signal_photons), ', mean_background_photon: ', ...
            num2str(mean_background_photons), ', SBR: ', num2str(SBR)]);

        alpha = albedo .* 1./ depth.^2; 
        signal_ppp = alpha ./ mean(alpha(:)) .* mean_signal_photons;
        ambient_ppp = (mean_background_photons) .* intensity ./ mean(intensity(:));

        % construct the inhomogeneous poisson process
        rates = zeros(bins1,bins2,num_bins);

        pulse_bins = (270e-12) / bin_size;
        pulse = normpdf(1:8*pulse_bins,(8*pulse_bins-1)/2,pulse_bins/2);

        rates(:,:,1:length(pulse)) = repmat(reshape(pulse, [1, 1, length(pulse)]),[bins1,bins2,1]);
        rates(:,:,1:length(pulse)) = rates(:,:,1:length(pulse)).*repmat(signal_ppp,[1,1,length(pulse)]);
        rates = rates + repmat(ambient_ppp./num_bins,[1,1,num_bins]);        

        % find amount to circshift the rate function
        [~, pulse_max_idx] = max(pulse);
        circ_amount = range_bins - pulse_max_idx;

        disp('sampling rate process');
        detections = zeros(size(rates));
        for jj = 1:bins1
            for kk = 1:bins2
                rates(jj,kk,:) = circshift(squeeze(rates(jj,kk,:)), circ_amount(jj,kk));
                detections(jj,kk,:) = poissrnd(rates(jj,kk,:));
            end
        end

        % sample the process
        spad = detections;
        clear detections;

        H = size(spad,1); W = size(spad,2);
        spad = reshape(spad, H*W, []);
        spad = sparse(spad); 

        % save sparse spad detections to file
        % the 'spad' is the simulated data, 'depth' is the GT 2D depth map,
        % which is 72*88, 'depth_hr' is 576*704. 'intensity' is the
        % gray image, 72*88. 'rates' is actually the GT 3D histogram.
        if LOW_RES
            out_fname = sprintf('%s/LR_%s_%s_%s.mat', outdir, scenes{ss}, num2str(mean_signal_photons), num2str(mean_background_photons));
        else
            out_fname = sprintf('%s/HR_%s_%s_%s.mat',outdir, scenes{ss}, num2str(mean_signal_photons), num2str(mean_background_photons));
        end
        save(out_fname, 'spad', 'depth', 'SBR', 'mean_signal_photons', 'mean_background_photons', 'bin_size','intensity','intensity_hr', 'depth_hr', 'range_bins');
    end
end
t_cost = toc(t_s);
disp(['Time cost: ', num2str(t_cost)]);

