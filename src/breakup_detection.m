% =========================================================================
% Script : jet_powerlaw_method1_batch_polygonROI.m
% Purpose:
%   METHOD 1: Power-law trajectory (batch over all images in a case folder)
%
%   NEW (Polygon ROI):
%     - Define polyVerts once (full-image coordinates)
%     - Build maskFull = poly2mask(...)
%     - Crop both image + mask with SAME cropBox
%     - Apply polygon mask inside ROI: I_crop(~maskCrop) = 0
%
%   Everything else unchanged:
%     - adaptive threshold + morphology
%     - nozzle-connected component
%     - power-law fit
%     - breakup detection
%     - diagnostic plots + CSVs + summary


clc; clear; close all;

%%  CPU PARALLEL SETUP (MINIMAL ADDITION) 
useParallelCPU = true;

% current local cluster is capped at 8 workers (from your error).
% Use 8 now. If you later increase the cap, you can set this to 16-18.
nWorkers = 8;

if useParallelCPU
    p = gcp('nocreate');
    if isempty(p)
        parpool('Processes', nWorkers);   % uses the "Processes" cluster
    else
        if p.NumWorkers ~= nWorkers
            delete(p);
            parpool('Processes', nWorkers);
        end
    end

    dq = parallel.pool.DataQueue;
    afterEach(dq, @(msg) fprintf('%s\n', msg));
else
    dq = [];
end

%%  USER INPUTS 

% --- Case folder containing the 100 spray images ---
caseFolder = '';

% --- Background image (single for this case) ---
bgPath = ''

% --- Nozzle pixel in FULL image (original coordinates, same for all 100) ---
nozzle_x_full = 134;
nozzle_y_full = 2943;

% --- Crop box (fixed) in FULL image coordinates ---
cropBox = [139 599 3929 2321];   % [x, y, width, height]

% --- Polygon vertices in FULL image coordinates (Nx2: [x y]) ---
% Example placeholder (REPLACE with your real polygon):
polyVerts = [139 2919; 271 2919; 279 2815; 4067 2755; 4063 599; 167 607; ];

% --- Calibration and nozzle diameter ---
px2mm = 0.010633;    % mm/pixel
d_mm  = 1.14;       % nozzle diameter (mm)

% --- Output directory inside the case folder ---
outDir = fullfile(caseFolder, 'results_powerlaw_method1');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

% --- Diagnostics saving behavior ---
saveDiagnosticsFirstOnly = true;   % true: only save PNGs for first image
                                   % false: save PNGs for every image

%%  TUNING PARAMETERS 

% Nozzle brightness ROI (fraction of image size)
noz_roi_y_start = 0.90;
noz_roi_x_end   = 0.50;
noz_gauss_sigma = 40;

% Adaptive core brightness parameters
core_y_start = 0.72;
core_x_end   = 0.18;

% Mild smoothing at end of preprocessing
final_smooth_sigma = 0.3;

% Global contrast stretch on I_corr
stretch_low  = 0.01;
stretch_high = 1.00;
gamma_val    = 1.0;

% Adaptive threshold for column mask
adaptive_sensitivity = 0.42;

% Morphology structuring element sizes
se_disk_main = 5;
min_area_pix = 5;

% Distance tolerance for breakup (in pixels)
tol_px = 2.0;

% Minimum number of points for power-law fit
min_fit_points = 10;

%%
% 0. COLLECT ALL SPRAY IMAGES IN CASE FOLDER
% 

sprayFiles = dir(fullfile(caseFolder, '*.JPG'));
if isempty(sprayFiles)
    error('No JPG files found in caseFolder: %s', caseFolder);
end

% Load background once
I_bg = im2double(imread(bgPath));

nImages = numel(sprayFiles);

% Preallocate arrays for summary
imageNames      = cell(nImages,1);
break_x_mm_vec  = NaN(nImages,1);
break_y_mm_vec  = NaN(nImages,1);
break_x_over_d  = NaN(nImages,1);
break_y_over_d  = NaN(nImages,1);
A_pow_vec       = NaN(nImages,1);
b_exp_vec       = NaN(nImages,1);
x0_mm_vec       = NaN(nImages,1);

%%
% MAIN LOOP OVER ALL IMAGES


% ---- ONLY CHANGE HERE: for -> parfor (everything inside unchanged) ----
parfor (i = 1:nImages, nWorkers)

    fname = sprayFiles(i).name;
    sprayPath = fullfile(caseFolder, fname);

    % ---- ONLY CHANGE: use DataQueue instead of fprintf for clean parallel output ----
    if useParallelCPU
        send(dq, sprintf('=== Processing image %d / %d: %s ===', i, nImages, fname));
    else
        fprintf('\n=== Processing image %d / %d: %s ===\n', i, nImages, fname);
    end

    imageNames{i} = fname;
    saveDiag = (~saveDiagnosticsFirstOnly && true) || (saveDiagnosticsFirstOnly && i==1);

    try
        %% 1. READ & PREPROCESS

        % 1.0 Read original spray image
        I_spray = im2double(imread(sprayPath));
        [h_full_color, w_full_color, ~] = size(I_spray);

        % Save original colour image (01)
        if saveDiag
            f0 = figure('Visible','off'); imshow(I_spray,[]);
            title('01 - Original colour image');
            saveas(f0, fullfile(outDir, sprintf('%02d_original.png',i)));
            close(f0);
        end

        % 1.1 Remove Magenta Reflections (LAB)
        lab = rgb2lab(I_spray);
        magentaMask = (lab(:,:,2) > 20) & (lab(:,:,3) < 5);
        magentaMask = imgaussfilt(double(magentaMask), 1) > 0.5;

        I_nomag = I_spray;
        for c = 1:3
            I_nomag(:,:,c) = regionfill(I_spray(:,:,c), magentaMask);
        end

        if saveDiag
            f_nomag = figure('Visible','off'); imshow(I_nomag,[]);
            title('02 - Magenta removed (colour)');
            saveas(f_nomag, fullfile(outDir, sprintf('%02d_nomag_color.png',i)));
            close(f_nomag);
        end

        % 1.2 Background Subtraction
        I_sub = I_nomag - I_bg;
        I_sub(I_sub < 0) = 0;

        % 1.3 Grayscale + Normalize
        I_gray = rgb2gray(I_sub);
        I_gray = I_gray - min(I_gray(:));
        I_gray = I_gray ./ max(I_gray(:));
        [h_full, w_full] = size(I_gray);

        maskFull = poly2mask(polyVerts(:,1), polyVerts(:,2), h_full, w_full);

        if saveDiag
            f1 = figure('Visible','off'); imshow(I_gray,[]);
            title('03 - Gray (magenta removed + BG subtracted)');
            saveas(f1, fullfile(outDir, sprintf('%02d_gray.png',i)));
            close(f1);

            f1b = figure('Visible','off'); imshow(maskFull);
            title('03b - Polygon mask (full image)');
            saveas(f1b, fullfile(outDir, sprintf('%02d_polyMask_full.png',i)));
            close(f1b);

            f1c = figure('Visible','off'); imshow(I_gray,[]); hold on;
            plot([polyVerts(:,1); polyVerts(1,1)], [polyVerts(:,2); polyVerts(1,2)], 'c-', 'LineWidth',1.5);
            title('03c - Polygon boundary on Gray (full)');
            saveas(f1c, fullfile(outDir, sprintf('%02d_polyBoundary_on_gray_full.png',i)));
            close(f1c);
        end

        % 1.4 Nozzle-region Brightness Correction
        roi = false(h_full, w_full);
        roi(round(h_full*noz_roi_y_start):end, 1:round(w_full*noz_roi_x_end)) = true;

        mean_jet   = mean(I_gray(roi));
        mean_total = mean(I_gray(:));

        if mean_jet < 0.8*mean_total
            gain = mean_total / max(mean_jet,1e-6);
            mask = imgaussfilt(double(roi), noz_gauss_sigma);
            I_corr = I_gray + mask .* (I_gray .* (gain - 1));
        else
            I_corr = I_gray;
        end
        I_corr = min(I_corr,1);

        % 1.5 Jet-core Adaptive Brightness Recovery
        roi_core = false(h_full, w_full);
        roi_core(round(h_full*core_y_start):end, 1:round(w_full*core_x_end)) = true;

        localMean = imgaussfilt(I_corr, 6);
        localVar  = imgaussfilt((I_corr - localMean).^2, 6);

        darkMask = roi_core & (I_corr < 0.7 * localMean);
        gainMap  = 1 + 0.8 * exp(-5 * localVar);
        gainMap  = imgaussfilt(gainMap, 10);

        I_boost = I_corr;
        I_boost(darkMask) = I_corr(darkMask) .* gainMap(darkMask);
        I_boost = imgaussfilt(I_boost, 0.8);
        I_boost = I_boost ./ max(I_boost(:));
        I_corr  = I_boost;

        % 1.6 Gentle smoothing + normalization
        I_corr = imgaussfilt(I_corr, final_smooth_sigma);
        I_corr = I_corr ./ max(I_corr(:));

        if saveDiag
            f2 = figure('Visible','off'); imshow(I_corr,[]);
            title('04 - I\_corr (brightness recovered)');
            saveas(f2, fullfile(outDir, sprintf('%02d_Icorr.png',i)));
            close(f2);
        end

        %% ---------- 1B. GLOBAL ENHANCEMENT ----------
        lowHigh = stretchlim(I_corr, [stretch_low stretch_high]);
        I_enh = imadjust(I_corr, lowHigh, []);
        I_enh = I_enh .^ gamma_val;
        I_enh = I_enh ./ max(I_enh(:));

        if saveDiag
            f3 = figure('Visible','off'); imshow(I_enh,[]);
            title('05 - I\_enh (enhanced, full image)');
            saveas(f3, fullfile(outDir, sprintf('%02d_Ienh_full.png',i)));
            close(f3);

            f3b = figure('Visible','off'); imshow(I_enh,[]); hold on;
            plot([polyVerts(:,1); polyVerts(1,1)], [polyVerts(:,2); polyVerts(1,2)], 'c-', 'LineWidth',1.5);
            plot(nozzle_x_full, nozzle_y_full, 'g+', 'MarkerSize',12, 'LineWidth',2);
            title('05b - Polygon boundary on I\_enh (full)');
            saveas(f3b, fullfile(outDir, sprintf('%02d_polyBoundary_on_Ienh_full.png',i)));
            close(f3b);
        end

        %% ---------- 2. CROP + APPLY POLYGON MASK ----------
        xC = cropBox(1);
        yC = cropBox(2);
        wC = cropBox(3);
        hC = cropBox(4);

        I_crop   = imcrop(I_enh, cropBox);
        maskCrop = imcrop(maskFull, cropBox);

        I_crop(~maskCrop) = 0;

        [h, w] = size(I_crop);

        nozzle_x_roi = nozzle_x_full - xC + 1;
        nozzle_y_roi = nozzle_y_full - yC + 1;

        if saveDiag
            f4a = figure('Visible','off'); imshow(I_crop,[]);
            title('06 - Cropped ROI (polygon-masked)');
            saveas(f4a, fullfile(outDir, sprintf('%02d_cropped_polygonMasked.png',i)));
            close(f4a);

            f4b = figure('Visible','off'); imshow(I_crop,[]); hold on;
            plot(nozzle_x_roi, nozzle_y_roi, 'g+', 'MarkerSize',10, 'LineWidth',1.5);
            title('07 - Cropped polygon ROI with nozzle');
            saveas(f4b, fullfile(outDir, sprintf('%02d_cropped_polygon_with_nozzle.png',i)));
            close(f4b);

            f4c = figure('Visible','off'); imshow(maskCrop);
            title('07b - Polygon mask (cropped)');
            saveas(f4c, fullfile(outDir, sprintf('%02d_polyMask_crop.png',i)));
            close(f4c);

            f4d = figure('Visible','off'); imshow(I_enh,[]); hold on;
            plot([polyVerts(:,1); polyVerts(1,1)], [polyVerts(:,2); polyVerts(1,2)], 'c-', 'LineWidth',1.5);
            rectangle('Position', cropBox, 'EdgeColor','y', 'LineWidth',1.5);
            plot(nozzle_x_full, nozzle_y_full, 'g+', 'MarkerSize',12, 'LineWidth',2);
            title('07c - Full I\_enh with polygon + cropBox');
            saveas(f4d, fullfile(outDir, sprintf('%02d_full_Ienh_polygon_cropBox.png',i)));
            close(f4d);
        end

        %% ---------- 3. COLUMN MASK ----------
        I_smooth2 = imgaussfilt(I_crop, 0.1);

        localThresh = imbinarize(I_smooth2, "adaptive", 'Sensitivity', adaptive_sensitivity);
        BW = I_smooth2 < localThresh;

        BW = imclose(BW, strel('disk', se_disk_main));
        BW = imfill(BW, 'holes');
        BW = bwareaopen(BW, min_area_pix);
        BW = imerode(BW, strel('disk', se_disk_main));
        BW = imclose(BW, strel('disk', se_disk_main));

        nozzle_x_roi = max(1, min(w, nozzle_x_roi));
        nozzle_y_roi = max(1, min(h, nozzle_y_roi));

        CC = bwconncomp(BW, 8);
        if CC.NumObjects == 0
            warning('No connected regions in BW for image %s. Skipping.', fname);
            continue;
        end

        labelMat = labelmatrix(CC);
        label_at_nozzle = labelMat(nozzle_y_roi, nozzle_x_roi);

        if label_at_nozzle == 0
            fprintf('No label exactly at nozzle. Searching nearest foreground...\n');
            [cols, rows] = meshgrid(1:w, 1:h);
            distMap = sqrt( (cols - nozzle_x_roi).^2 + (rows - nozzle_y_roi).^2 );
            distMap(~BW) = Inf;
            [~, idx_min] = min(distMap(:));
            if isinf(distMap(idx_min))
                warning('No foreground near nozzle for image %s. Skipping.', fname);
                continue;
            end
            [y_near, x_near] = ind2sub([h, w], idx_min);
            label_at_nozzle = labelMat(y_near, x_near);
            fprintf('Using nearest foreground at (%d,%d), label %d.\n', x_near, y_near, label_at_nozzle);
        end

        BW_col = (labelMat == label_at_nozzle);

        if saveDiag
            f5 = figure('Visible','off'); imshow(I_crop,[]); hold on;
            contour(BW_col, [0.5 0.5], 'r', 'LineWidth',1);
            plot(nozzle_x_roi, nozzle_y_roi, 'g+', 'MarkerSize',10,'LineWidth',1.5);
            title('08 - Column boundary on cropped polygon ROI');
            saveas(f5, fullfile(outDir, sprintf('%02d_BW_col_on_crop.png',i)));
            close(f5);
        end

        %% ---------- FULL IMAGE OVERLAYS ----------
        BW_full = false(h_full, w_full);
        row_idx = yC:(yC + h - 1);
        col_idx = xC:(xC + w - 1);
        BW_full(row_idx, col_idx) = BW_col;

        if saveDiag
            f6 = figure('Visible','off'); imshow(I_enh,[]); hold on;
            contour(BW_full, [0.5 0.5], 'r', 'LineWidth',1);
            plot([polyVerts(:,1); polyVerts(1,1)], [polyVerts(:,2); polyVerts(1,2)], 'c-', 'LineWidth',1.2);
            title('09 - Column boundary on I\_enh (full) + polygon boundary');
            saveas(f6, fullfile(outDir, sprintf('%02d_BW_col_on_Ienh.png',i)));
            close(f6);
        end

        %% ---------- 4. COLUMN PIXELS → CARTESIAN → mm ----------
        [y_img, x_img] = find(BW_col);
        x_cart_px = x_img;
        y_cart_px = h - y_img + 1;

        x_mm = x_cart_px * px2mm;
        y_mm = y_cart_px * px2mm;

        %% ---------- 5. POWER-LAW FIT & BREAKUP ----------
        x0_px = min(x_cart_px);
        x0_mm = x0_px * px2mm;

        x_rel_px = x_cart_px - x0_px + 1;
        x_rel_mm = x_rel_px * px2mm;

        valid = (x_rel_mm > 0) & (y_mm > 0);

        if nnz(valid) < min_fit_points
            warning('Not enough points for power-law fit in image %s. Skipping fit.', fname);
            A_pow = NaN; b_exp = NaN;
            x_mm_curve = []; y_mm_curve = [];
            break_x_mm = NaN; break_y_mm = NaN;
        else
            X = log(x_rel_mm(valid));
            Y = log(y_mm(valid));

            p     = polyfit(X, Y, 1);
            b_exp = p(1);
            A_pow = exp(p(2));

            x_rel_mm_curve = linspace(min(x_rel_mm(valid)), max(x_rel_mm(valid))*1.15, 600);
            y_mm_curve     = A_pow * (x_rel_mm_curve.^b_exp);
            x_mm_curve     = x0_mm - px2mm + x_rel_mm_curve;

            x_cart_px_curve = x_mm_curve / px2mm;
            y_cart_px_curve = y_mm_curve / px2mm;

            D = bwdist(~BW_col);

            break_x_px = NaN; break_y_px = NaN;

            for k2 = numel(x_cart_px_curve):-1:1
                xi_cart = x_cart_px_curve(k2);
                yi_cart = y_cart_px_curve(k2);

                xi = round(xi_cart);
                yi = round(yi_cart);
                if xi < 1 || xi > w || yi < 1 || yi > h
                    continue;
                end

                y_img_here = h - yi + 1;
                if y_img_here < 1 || y_img_here > h
                    continue;
                end

                if D(y_img_here, xi) >= tol_px
                    break_x_px = xi_cart;
                    break_y_px = yi_cart;
                    break;
                end
            end

            if ~isnan(break_x_px)
                break_x_mm = break_x_px * px2mm;
                break_y_mm = break_y_px * px2mm;
            else
                break_x_mm = NaN;
                break_y_mm = NaN;
            end
        end

        %% ---------- 10 - FULL OVERLAY (BOUNDARY + FIT + BREAKUP) ----------
        if ~isempty(x_mm_curve) && saveDiag
            x_curve_px_roi = x_mm_curve / px2mm;
            y_curve_px_roi = y_mm_curve / px2mm;

            row_curve_roi = h - y_curve_px_roi + 1;
            col_curve_roi = x_curve_px_roi;

            col_curve_full = xC + col_curve_roi - 1;
            row_curve_full = yC + row_curve_roi - 1;

            f7 = figure('Visible','off'); imshow(I_enh,[]); hold on;
            contour(BW_full, [0.5 0.5], 'r', 'LineWidth',1);
            plot(col_curve_full, row_curve_full, 'y-', 'LineWidth',1.5);

            if ~isnan(break_x_mm)
                bx_roi = break_x_mm / px2mm;
                by_roi = break_y_mm / px2mm;
                by_row_roi = h - by_roi + 1;
                bx_full = xC + bx_roi - 1;
                by_full = yC + by_row_roi - 1;
                plot(bx_full, by_full, 'go', 'MarkerSize',8, 'LineWidth',1.5);
            end

            title('10 - Full: column boundary + fit + breakup + polygon');
            saveas(f7, fullfile(outDir, sprintf('%02d_fullOverlay_fit.png',i)));
            close(f7);
        end

        %% ---------- 6. UPDATE SUMMARY ARRAYS ----------
        A_pow_vec(i)      = A_pow;
        b_exp_vec(i)      = b_exp;
        x0_mm_vec(i)      = x0_mm;
        break_x_mm_vec(i) = break_x_mm;
        break_y_mm_vec(i) = break_y_mm;
        break_x_over_d(i) = break_x_mm / d_mm;
        break_y_over_d(i) = break_y_mm / d_mm;

        %% ---------- 7. WRITE CSVS FOR THIS IMAGE ----------
        [~, baseName, ~] = fileparts(fname);

        T_raw = table(x_mm(:), y_mm(:), x_mm(:)/d_mm, y_mm(:)/d_mm, ...
            'VariableNames', {'x_mm','y_mm','x_over_d','y_over_d'});
        writetable(T_raw, fullfile(outDir, [baseName '_rawTrajectory.csv']));

        if ~isempty(x_mm_curve)
            T_fit = table(x_mm_curve(:), y_mm_curve(:), x_mm_curve(:)/d_mm, y_mm_curve(:)/d_mm, ...
                'VariableNames', {'x_mm','y_mm','x_over_d','y_over_d'});
            writetable(T_fit, fullfile(outDir, [baseName '_fitCurve.csv']));
        end

        T_break = table(break_x_mm, break_y_mm, break_x_mm/d_mm, break_y_mm/d_mm, ...
            'VariableNames', {'break_x_mm','break_y_mm','break_x_over_d','break_y_over_d'});
        writetable(T_break, fullfile(outDir, [baseName '_breakup.csv']));

        T_eq = table(A_pow, b_exp, x0_mm, 'VariableNames', {'A_pow','b_exp','x0_mm'});
        writetable(T_eq, fullfile(outDir, [baseName '_equation.csv']));

        %% ---------- 11/12. MM-SPACE PLOTS ----------
        if saveDiag
            f8 = figure('Visible','off');
            scatter(x_mm, y_mm, 4, 'b', 'filled'); hold on; box on;
            xlabel('x (mm)'); ylabel('y (mm)');
            title(sprintf('%02d - Mask (mm)', i));
            set(gca,'YDir','normal');
            saveas(f8, fullfile(outDir, sprintf('%02d_mask_mm.png',i)));
            close(f8);

            f9 = figure('Visible','off');
            scatter(x_mm, y_mm, 4, 'b', 'filled'); hold on; box on;
            if ~isempty(x_mm_curve)
                plot(x_mm_curve, y_mm_curve, 'r-', 'LineWidth',1.5);
            end
            if ~isnan(break_x_mm)
                plot(break_x_mm, break_y_mm, 'go','MarkerSize',8,'LineWidth',1.5);
            end
            xlabel('x (mm)'); ylabel('y (mm)');
            title(sprintf('%02d - Mask + Fit + Breakup (mm)', i));
            set(gca,'YDir','normal');
            saveas(f9, fullfile(outDir, sprintf('%02d_mask_fit_breakup_mm.png',i)));
            close(f9);
        end

    catch ME
        warning('Error processing image %s: %s', fname, ME.message);
        continue;
    end
end


% FINAL SUMMARY OVER ALL IMAGES


summaryTable = table(imageNames, ...
    break_x_mm_vec, break_y_mm_vec, ...
    break_x_over_d, break_y_over_d, ...
    A_pow_vec, b_exp_vec, x0_mm_vec, ...
    'VariableNames', {'imageName','break_x_mm','break_y_mm', ...
                      'break_x_over_d','break_y_over_d', ...
                      'A_pow','b_exp','x0_mm'});

avg_break_x_mm = mean(break_x_mm_vec, 'omitnan');
avg_break_y_mm = mean(break_y_mm_vec, 'omitnan');
avg_A_pow      = mean(A_pow_vec,      'omitnan');
avg_b_exp      = mean(b_exp_vec,      'omitnan');
avg_x0_mm      = mean(x0_mm_vec,      'omitnan');

summaryTable(end+1,:) = { ...
    'AVERAGE', ...
    avg_break_x_mm, avg_break_y_mm, ...
    avg_break_x_mm/d_mm, avg_break_y_mm/d_mm, ...
    avg_A_pow, avg_b_exp, avg_x0_mm};

writetable(summaryTable, fullfile(outDir, 'summary_all_images.csv'));

fprintf('\n=== BATCH PROCESSING COMPLETE ===\n');
fprintf('Summary CSV saved to: %s\n', fullfile(outDir,'summary_all_images.csv'));
fprintf('Results folder: %s\n', outDir);

