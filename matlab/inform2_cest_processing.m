% ================================
%
% Author: Manus J. Donahue
% Purpose: Read dicom files for INFORM2 CEST data, parse, and save as
% .nii.gz with correct orientation for fslview
% Date: 14 Dec 2020
%
% Dependencies, dicomread (Matlab), st2, save_avw (FSL)
%
% Will need to change the path for input and output data (% path)
%
% ================================

tic();

preproc_cest=1;

% ======================================================================
%
% Read in and save the unprocessed CEST data
%
% ======================================================================



%basedirs = ["/Users/skyjones/Desktop/cest_processing/data/working/aim4","/Users/skyjones/Desktop/cest_processing/data/working/aim3",...
%"/Users/skyjones/Desktop/cest_processing/data/working/aim2","/Users/skyjones/Desktop/cest_processing/data/working/aim1"];
basedirs = ["/Users/skyjones/Desktop/hiv_processing/data/working/aim5"];
%basedirs = ["/Users/skyjones/Desktop/cest_processing/data/working/aim1"];

%basedirs = ["/Users/skyjones/Desktop/cest_processing/data/working/aim4","/Users/skyjones/Desktop/cest_processing/data/working/aim3",...
%"/Users/skyjones/Desktop/cest_processing/data/working/aim2","/Users/skyjones/Desktop/cest_processing/data/working/aim1",...
%"/Users/skyjones/Desktop/hiv_processing/data/working/aim5"];

disp('Initializing')


for kk=1:(size(basedirs,2))
    basedir = basedirs(kk);
    disp(basedir)
    
    basedir = convertStringsToChars(basedir);
    
    %weight_types = ["water", "fat"];
    weight_types = ["water"];

    mydir = [basedir '/raw'];
    outdir= [basedir '/processed']; % note these are being saved locally


    myfiles=dir(mydir);

    good_validated = [];
    bad_validated = [];
    val_file=strcat(outdir,'/','validation.txt');

    processing_problems = [];
    problem_file=strcat(outdir,'/','problem_files.txt');
    
    for ff=3:(size(myfiles,1)) % note that . and .. are the first two files
    %save_avw(dd(:,:,1),file2save,'d',[1.8 2 6 7.692]); % path


        overwrite = false;
        the_name = myfiles(ff).name;
        the_split = strsplit(the_name, '_');
        the_num = str2double(the_split{2});
        the_num_2 = str2double(the_split{3});

        if strcmp(the_name, '.DS_Store')
            continue;
        end

        lastdot_pos = find(the_name == '.', 1, 'last');
        clean_name = the_name(1:lastdot_pos-1);

        newfolder = strcat(outdir,'/',clean_name);

        if and(exist(newfolder, 'dir'), ~overwrite)
            disp('data has been processed and overwrite is off')
            continue
        end


        mkdir(newfolder);

        file2save=strcat(newfolder,'/',clean_name);

        if preproc_cest==1
            disp('file is:')
            disp(strcat(mydir,'/',myfiles(ff).name))
            dd=squeeze(dicomread(strcat(mydir,'/',myfiles(ff).name)));
            % read data

            % slices = 10;
            % dynamics = 50;

            try
                if (the_num < 140463 || the_num == 141366) && the_num ~= 139423 && ~(the_num == 139365 && the_num_2 == 2) && ~(the_num == 139285 && the_num_2 == 2) % pts after this number use an alternate encoding. I don't know why there are random exceptions
                    disp('Standard encoding')
                    cc1=dd(:,:,1:500); % water
                    cc2=dd(:,:,501:1000); % f+w in phase
                    cc3=dd(:,:,1001:1500); % f+w out of phase
                    cc4=dd(:,:,1501:2000); % fat
                    cc5=dd(:,:,2001:2500); % T2*
                    cc6=dd(:,:,2501:3000); % b0
                else
                    disp('Alternate encoding')
                    cc6=dd(:,:,1:500); % interleaved f+w in phase, f+w out of phase, and something that looks like f+w out of phase
                    cc2=dd(:,:,501:1000); % interleaved f+w in phase, f+w out of phase, and something that looks like f+w out of phase
                    cc3=dd(:,:,1001:1500); % interleaved f+w in phase, f+w out of phase, and something that looks like f+w out of phase
                    cc1=dd(:,:,1501:2000); % water
                    cc4=dd(:,:,2001:2500); % fat
                    cc5=dd(:,:,2501:3000); % T2*
                end
            catch
                rmdir(newfolder)
                disp('Problem processing (likely index out of bounds)')
                processing_problems = [processing_problems '\n' clean_name];
                continue
            end



            cest1=reshape(cc1,[80 80 50 10]); % water
            cest2=reshape(cc2,[80 80 50 10]); % f+w in phase
            cest3=reshape(cc3,[80 80 50 10]); % f+w out of phase
            cest4=reshape(cc4,[80 80 50 10]); % fat
            cest5=reshape(cc5,[80 80 50 10]); % T2*
            cest6=reshape(cc6,[80 80 50 10]); % b0

            cest1=permute(cest1,[2 1 4 3]);
            cest2=permute(cest2,[2 1 4 3]);
            cest3=permute(cest3,[2 1 4 3]);
            cest4=permute(cest4,[2 1 4 3]);
            cest5=permute(cest5,[2 1 4 3]);
            cest6=permute(cest6,[2 1 4 3]);

            for nd=1:size(cest1,4)
                dummy1=squeeze(cest1(:,:,:,nd));
                dummy2=squeeze(cest2(:,:,:,nd));
                dummy3=squeeze(cest3(:,:,:,nd));
                dummy4=squeeze(cest4(:,:,:,nd));
                dummy5=squeeze(cest5(:,:,:,nd));
                dummy6=squeeze(cest6(:,:,:,nd));

                cest1save(:,:,:,nd)=(dummy1);
                cest2save(:,:,:,nd)=(dummy2);
                cest3save(:,:,:,nd)=(dummy3);
                cest4save(:,:,:,nd)=(dummy4);
                cest5save(:,:,:,nd)=(dummy5);
                cest6save(:,:,:,nd)=(dummy6);

                clear dummy1;
                clear dummy2;
                clear dummy3;
                clear dummy4;
                clear dummy5;
                clear dummy6;
            end

            % some rough data validation - the biggest threshholded shape in the
            % fat image should be a big arc, the biggest threshholded shape in
            % the water image should be a big blob. So, the fat image's biggest
            % shape should have lower geometric solidity than the water image's

            SE = strel([1 1; 1 1]);

            d_water = double(cest1save(:,:,7,15));
            n_water = normalize(d_water);
            bw_water = imbinarize(n_water);
            bw_water= imclose(bw_water, SE);

            d_fat = double(cest4save(:,:,7,15));
            n_fat = normalize(d_fat);
            bw_fat = imbinarize(n_fat);
            bw_fat = imclose(bw_fat, SE);

            %figure
            %imshow(n_water, [-2 2], 'InitialMagnification', 800)
            %figure
            %imshow(bw_water, [0 1], 'InitialMagnification', 800)

            %[centers_water, radii_water] = imfindcircles(n_water, [4 12], 'ObjectPolarity','dark', 'Sensitivity', 0.9); % , 'EdgeThreshold', 0.2);
            %h_water = viscircles(centers_water,radii_water);

            %figure
            %imshow(n_fat, [-2 2], 'InitialMagnification', 800)
            %figure
            %imshow(bw_fat, [0 1], 'InitialMagnification', 800)

            %[centers_fat, radii_fat] = imfindcircles(n_fat, [3 9], 'ObjectPolarity','bright', 'Sensitivity', 0.85); % , 'EdgeThreshold', 0.2);
            %h_fat = viscircles(centers_fat,radii_fat);

            CC_water = bwconncomp(bw_water);
            CC_fat = bwconncomp(bw_fat);

            areas_water = regionprops(CC_water, 'Area');
            areas_fat = regionprops(CC_fat, 'Area');

            solidities_water = regionprops(CC_water, 'Solidity');
            solidities_fat = regionprops(CC_fat, 'Solidity');

            [max_area_water, max_area_idx_water] = max(struct2array(areas_water));
            [max_area_fat, max_area_idx_fat] = max(struct2array(areas_fat));

            area_water = areas_water(max_area_idx_water).Area;
            area_fat = areas_fat(max_area_idx_fat).Area;

            solid_water = solidities_water(max_area_idx_water).Solidity;
            solid_fat = solidities_fat(max_area_idx_fat).Solidity;

            if ~(solid_water > solid_fat)
                disp('Imaging indices may be out of order (water shape solidity should be greater than fat shape solidity)')
                bad_validated = [bad_validated '\n\t' clean_name];
                txtname = strcat(file2save,'_badstatus.txt');
                msg = ['Imaging indices may be out of order\nwater area,solidity: ' num2str(area_water) ', ' num2str(solid_water) ...
                    '\nfat area,solidity: ' num2str(area_fat) ', ' num2str(solid_fat)];
            else
                good_validated = [good_validated '\n\t' clean_name];
                txtname = strcat(file2save,'_goodstatus.txt');
                           msg = ['Imaging indices check out\nwater area,solidity: ' num2str(area_water) ', ' num2str(solid_water) ...
                    '\nfat area,solidity: ' num2str(area_fat) ', ' num2str(solid_fat)];
            end


            fid = fopen(txtname,'wt');
            fprintf(fid, msg);
            fclose(fid);



            % WILL NEED TO CHANGE PATH FOR YOUR COMPUTER
            save_avw(cest1save,strcat(file2save,'_C1_'),'d',[1.8 2 6 7.692]); % path
            save_avw(cest2save,strcat(file2save,'_C2_'),'d',[1.8 2 6 7.692]); % path
            save_avw(cest3save,strcat(file2save,'_C3_'),'d',[1.8 2 6 7.692]); % path
            save_avw(cest4save,strcat(file2save,'_C4_'),'d',[1.8 2 6 7.692]); % path
            save_avw(cest5save,strcat(file2save,'_C5_'),'d',[1.8 2 6 7.692]); % path
            save_avw(cest6save,strcat(file2save,'_C6_'),'d',[1.8 2 6 7.692]); % path

        end

        disp('Saved raw data')

        % ======================================================================
        %
        % End read in the CEST data
        %
        % ======================================================================

        % ======================================================================
        %
        % Process the CEST data and save the processed maps
        %
        % ======================================================================

                for jj=1:(size(weight_types,2))

                    weight_txt = weight_types(jj);

                    disp('On weighting:');
                    disp(weight_txt);

                    if weight_txt == 'water'
                        disp('using water')
                        the_cest_weight = cest1;
                    elseif weight_txt == 'fat'
                        disp('using fat')
                        the_cest_weight = cest4;
                    else
                        throw(MException('MYFUN:incorrect','incorrect weighting specification.'))
                    end

                    
                    
                    weight_txt = convertStringsToChars(weight_txt);

                    cest1_eq=mean(the_cest_weight(:,:,:,[1,2,3,48,49,50]),4); % first 3 and last 3 are equilibrium
                    cest1_z=the_cest_weight(:,:,:,4:47); % all but first and last 3 have offsets
                    
                    cest1_z_water = cest1(:,:,:,4:47); % we will always use the water-weighted image for b0 correction

                    % The pre-pulse offset sampling is not symmetric or uniform, so need to do the interpolation piece-wise (actual offsets included at bottom of file)
                    % Note we will re-sample to 0.1 ppm -----------------------------------
                    ppm_interp=-5.5:0.1:5.5; % our desired ppm sampling after spline interpolation
                    cest1_z_forb0correct=zeros(size(cest1_z,1), size(cest1_z,2), size(cest1_z,3),size(ppm_interp,2)); % empty z-spectrum matrix to populate
                    
                    cest1_z_forb0correct_water=zeros(size(cest1_z_water,1), size(cest1_z_water,2), size(cest1_z_water,3),size(ppm_interp,2)); % empty z-spectrum matrix to populate

                    for nz=1:size(cest1_z,3)
                        nz,
                        for nx=1:size(cest1_z,1)
                            for ny=1:size(cest1_z,2)

                                % region 1:
                                ri1=[1:6];
                                tempz=squeeze(cest1_z(nx,ny,nz,ri1));
                                tempz_water=squeeze(cest1_z_water(nx,ny,nz,ri1));
                                % this part is sampled at 0.4 ppm, so need to resample 4-fold
                                % higher:
                                xxxri1=1:0.25:6; % note we use the absolute indices for spline
                                
                                yyyr1=spline(ri1,double(tempz),xxxri1); % this does the spline
                                cest1_z_forb0correct(nx,ny,nz,1:size(xxxri1,2))=yyyr1; % put into the resampled cest map
                                clear tempz;
                                yyyr1_water=spline(ri1,double(tempz_water),xxxri1); % this does the spline
                                cest1_z_forb0correct_water(nx,ny,nz,1:size(xxxri1,2))=yyyr1_water; % put into the resampled cest map
                                clear tempz_water;

                                % region 2:
                                ri2=[6:7];
                                tempz=squeeze(cest1_z(nx,ny,nz,ri2));
                                tempz_water=squeeze(cest1_z_water(nx,ny,nz,ri2));
                                % this part is sampled at 0.5 ppm, so need to resample 5-fold
                                % higher
                                xxxri2=1:0.2:2; % note just two ppm values (-3.5 to -3.0)
                                
                                yyyr2=spline(1:size(ri2,2),double(tempz),xxxri2);
                                cest1_z_forb0correct(nx,ny,nz,size(xxxri1,2):(size(xxxri1,2)+size(xxxri2,2)-1))=yyyr2;
                                clear tempz;
                                yyyr2_water=spline(1:size(ri2,2),double(tempz_water),xxxri2);
                                cest1_z_forb0correct_water(nx,ny,nz,size(xxxri1,2):(size(xxxri1,2)+size(xxxri2,2)-1))=yyyr2_water;
                                clear tempz_water;

                                % region 3:
                                ri3=[7:12];
                                tempz=squeeze(cest1_z(nx,ny,nz,ri3));
                                tempz_water=squeeze(cest1_z_water(nx,ny,nz,ri3));
                                % this part is sampled at 0.4 ppm, so need to resample 4-fold
                                % higher:
                                xxxri3=1:0.25:6; % note we use the absolute indices for spline
                                
                                yyyr3=spline(1:size(ri3,2),double(tempz),xxxri3);
                                cest1_z_forb0correct(nx,ny,nz,(size(xxxri1,2)+size(xxxri2,2)-1):((size(xxxri1,2)+size(xxxri2,2)-1)+size(xxxri3,2)-1))=yyyr3;
                                clear tempz;
                                yyyr3_water=spline(1:size(ri3,2),double(tempz_water),xxxri3);
                                cest1_z_forb0correct_water(nx,ny,nz,(size(xxxri1,2)+size(xxxri2,2)-1):((size(xxxri1,2)+size(xxxri2,2)-1)+size(xxxri3,2)-1))=yyyr3_water;
                                clear tempz;
                                endindex3=((size(xxxri1,2)+size(xxxri2,2)-1)+size(xxxri3,2)-1);

                                % region 4:
                                ri4=[12:27]; % resample this 2-fold (0.2 ppm to 0.1 ppm)
                                tempz=squeeze(cest1_z(nx,ny,nz,ri4));
                                tempz_water=squeeze(cest1_z_water(nx,ny,nz,ri4));
                                % this part is sampled at 0.2 ppm, so need to resample 2-fold
                                % higher:
                                xxxri4=1:0.5:16; % note we use the absolute indices for spline
                                
                                yyyr4=spline(1:size(ri4,2),double(tempz),xxxri4);
                                cest1_z_forb0correct(nx,ny,nz,endindex3:(endindex3+size(xxxri4,2)-1))=yyyr4;
                                clear tempz;
                                yyyr4_water=spline(1:size(ri4,2),double(tempz_water),xxxri4);
                                cest1_z_forb0correct_water(nx,ny,nz,endindex3:(endindex3+size(xxxri4,2)-1))=yyyr4_water;
                                clear tempz;
                                endindex4=(endindex3+size(xxxri4,2)-1);

                                % region 5:
                                ri5=[27:28]; % resample 3-fold (0.3 ppm to 0.1 ppm)
                                tempz=squeeze(cest1_z(nx,ny,nz,ri5));
                                tempz_water=squeeze(cest1_z_water(nx,ny,nz,ri5));
                                % this part is sampled at 0.3 ppm, so need to resample 3-fold
                                % higher:
                                xxxri5=1:0.33333:2; % note we use the absolute indices for spline
                                
                                yyyr5=spline(1:size(ri5,2),double(tempz),xxxri5);
                                cest1_z_forb0correct(nx,ny,nz,endindex4:(endindex4+size(xxxri5,2)-1))=yyyr5;
                                clear tempz;
                                yyyr5_water=spline(1:size(ri5,2),double(tempz_water),xxxri5);
                                cest1_z_forb0correct_water(nx,ny,nz,endindex4:(endindex4+size(xxxri5,2)-1))=yyyr5_water;
                                clear tempz;
                                endindex5=(endindex4+size(xxxri5,2)-1);

                                % region 6:
                                ri6=[28:44]; % resample this 2-fold (0.2 ppm to 0.1 ppm)
                                tempz=squeeze(cest1_z(nx,ny,nz,ri6));
                                tempz_water=squeeze(cest1_z_water(nx,ny,nz,ri6));
                                % this part is sampled at 0.2 ppm, so need to resample 2-fold
                                % higher:
                                xxxri6=1:0.5:17; % note we use the absolute indices for spline
                                
                                yyyr6=spline(1:size(ri6,2),double(tempz),xxxri6);
                                cest1_z_forb0correct(nx,ny,nz,endindex5:(endindex5+size(xxxri6,2)-1))=yyyr6;
                                clear tempz;
                                yyyr6_water=spline(1:size(ri6,2),double(tempz_water),xxxri6);
                                cest1_z_forb0correct_water(nx,ny,nz,endindex5:(endindex5+size(xxxri6,2)-1))=yyyr6_water;
                                clear tempz;
                                endindex6=(endindex5+size(xxxri6,2)-1);

                            end
                        end
                    end
                    % Above is the end of the piece-wise interpolation --------------------

                    % save a copy of data before doing b0 correction
                    cest1_z_uncorrected = cest1_z_forb0correct;

                    % now do the b0 correction --------------------------------------------
                    I=zeros(size(cest1_z,1),size(cest1_z,2), size(cest1_z,3));
                    Y=I;
                    for nz=1:size(cest1_z,3)
                        nz,
                        for nx=1:size(cest1_z,1)
                            for ny=1:size(cest1_z,2)
                                [Y(nx,ny,nz) I(nx,ny,nz)]=(min(squeeze(cest1_z_forb0correct_water(nx,ny,nz,:)))); % I is the index of the minimum; Y is the value of this
                            end
                        end
                    end
                    % calculate the b0 shift map in units of ppm
                    b0=(I-56).*0.1; % because 56 is the center index of the interpolated z-spectrum and 0.1 ppm is the increment of the interpolated spectrum

                    % Calculate the ppm shift; note index=56 is 0 ppm, and everything is in
                    % 0.1 ppm increments
                    cest1_z_b0correct=zeros(size(cest1_z,1), size(cest1_z,2), size(cest1_z,3),81); % the B0-corrected z-spectrum to populate
                    for nz=1:size(cest1_z,3)
                        nz,
                        for nx=1:size(cest1_z,1)
                            for ny=1:size(cest1_z,2)
                                if (I(nx,ny,nz)>41 & I(nx,ny,nz)<71) % here we are excluding everything outside of 1.5 ppm B0 shift
                                    cest1_z_b0correct(nx,ny,nz,41)=cest1_z_forb0correct(nx,ny,nz,I(nx,ny,nz)); % the center frequency
                                    cest1_z_b0correct(nx,ny,nz,42:81)=cest1_z_forb0correct(nx,ny,nz,(I(nx,ny,nz)+1):(I(nx,ny,nz)+1+39));
                                    cest1_z_b0correct(nx,ny,nz,1:40)=cest1_z_forb0correct(nx,ny,nz,(I(nx,ny,nz)-1-39):(I(nx,ny,nz)-1));
                                end
                            end
                        end
                    end
                    % end of the b0 correction --------------------------------------------

                    % calcualate the APT, NOE (-APT), CHO, B0, Equilibrium, and asymmetry
                    % maps ----------------------------------------------------------------

                    % these are the APT indices of the -4 to +4 ppm spectrum.
                    % note that so far this is just as acquired, so -ppm is on the left and
                    % +ppm on the right (opposite of convention)

                    ppm_after_correct=-4:0.1:4; % this is the ppm range after throwing out everything with more than a +/- 1.5 ppm shift

                    aptleft=[4:8]; % -3.3 to -3.7 ppm; note this is about NOE
                    aptright=[74:78]; % +3.3 to +3.7 ppm; about the APT range
                    
                    %truenoe=[6:26]; % -1.5 to -3.5 ppm
                    truenoe=[11:21]; % -2 to -3 ppm

                    % now calculate the CHO (choline). Note this is on the minus end of the spectrum,
                    % and since thisn't flipped to have +ppm on the left yet, the CHO
                    % effect will be on the left side here and it's centered at -1.6 ppm.
                    choleft=[21:29]; % we're averaging -1.2 to -2 ppm here
                    choright=[53:61]; % we're averaging 1.2 to 2 ppm here
                    
                    truenoe_cest1_z_b0corrected=mean(cest1_z_b0correct(:,:,:,truenoe),4);

                    apt_left_cest1_z_b0corrected=mean(cest1_z_b0correct(:,:,:,aptleft),4); % this is -ppm
                    apt_right_cest1_z_b0corrected=mean(cest1_z_b0correct(:,:,:,aptright),4); % this is +ppm

                    cho_left_cest1_z_b0corrected=mean(cest1_z_b0correct(:,:,:,choleft),4); % this is -ppm
                    cho_right_cest1_z_b0corrected=mean(cest1_z_b0correct(:,:,:,choright),4); % this is +ppm

                    disp('SAVING');
                    % ================================================
                    % save the relevant CEST maps:
                    % The mean signal with pre-pulse far from water resonance (CEST_S0):
                    save_avw(cest1_eq,strcat(file2save,'OCEST_S0_', weight_txt),'d',[1.8 2 6 7.692]); % path
                    %save_avw(st2(cest1_eq),'/Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_S0','d',[1.8 2 6 7.692]); % path
                    %unix('fslswapdim /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_S0.nii.gz -x y z /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_S0.nii.gz'); % path

                    % the uncorrected cest map
                    save_avw(cest1_z_uncorrected,strcat(file2save,'OCEST_uncorrected_', weight_txt),'d',[1.8 2 6 7.692]); % path

                    % the b0 shift map (in units of ppm)
                    save_avw(b0,strcat(file2save,'OCEST_B0_', weight_txt),'d',[1.8 2 6 7.692]); % path
                    %save_avw(st2(b0),'/Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_B0','d',[1.8 2 6 7.692]); % path
                    %unix('fslswapdim /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_B0.nii.gz -x y z /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_B0.nii.gz'); % path
                    
                    %NOE:
                    % The mean signal between -1.5 ppm and -3.5ppm (NOE)
                    save_avw(truenoe_cest1_z_b0corrected,strcat(file2save,'OCEST_NOE_', weight_txt),'d',[1.8 2 6 7.692]); % path
                    
                    %APT:
                    % The mean signal between -3.3 ppm and -3.7ppm (OPPAPT):
                    save_avw(apt_left_cest1_z_b0corrected,strcat(file2save,'OCEST_OPPAPT_', weight_txt),'d',[1.8 2 6 7.692]); % path
                    %save_avw(st2(apt_left_cest1_z_b0corrected),'/Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_OPPAPT','d',[1.8 2 6 7.692]); % path
                    %unix('fslswapdim /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_OPPAPT.nii.gz -x y z /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_OPPAPT.nii.gz'); % path


                    % The mean signal at +3.3 ppm and +3.7ppm (APT):
                    save_avw(apt_right_cest1_z_b0corrected,strcat(file2save,'OCEST_APT_', weight_txt),'d',[1.8 2 6 7.692]); % path
                    %save_avw(st2(apt_right_cest1_z_b0corrected),'/Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_APT','d',[1.8 2 6 7.692]); % path
                    %unix('fslswapdim /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_APT.nii.gz -x y z /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_APT.nii.gz'); % path

                    % The mean APT asymmetry map [(+3.3 ppm and +3.7ppm) - (-3.3 ppm and -3.7ppm)] / (S0) (APTASYM):
                    % NOTE: ASYMMETRY NEEDS TO BE CALCULATED IN BASH
                    %unix('fslmaths /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_APT.nii.gz -sub /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_OPPAPT.nii.gz -div /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_S0.nii.gz /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_APTASYM.nii.gz'); % path
                    % End APT

                    % CHO:
                    % The mean signal between -1.2 ppm and -2 ppm (CHO; center on -1.6 ppm):
                    save_avw(cho_left_cest1_z_b0corrected,strcat(file2save,'OCEST_CHO_', weight_txt),'d',[1.8 2 6 7.692]); % path
                    %unix('fslswapdim /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_CHO.nii.gz -x y z /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_CHO.nii.gz'); % path

                    % The mean signal at +1.2 ppm and +2 ppm (OPPCHO):
                    save_avw(cho_right_cest1_z_b0corrected,strcat(file2save,'OCEST_OPPCHO_', weight_txt),'d',[1.8 2 6 7.692]); % path
                    %unix('fslswapdim /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_OPPCHO.nii.gz -x y z /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_OPPCHO.nii.gz'); % path

                    % The mean CHO asymmetry map [(-2 ppm and -1.2 ppm) - (+1.2 ppm and +2 ppm)] / (S0) (NOEASYM):
                    % NOTE: ASYMMETRY NEEDS TO BE CALCULATED IN BASH
                    %unix('fslmaths /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_CHO.nii.gz -sub /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_OPPCHO.nii.gz -div /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_S0.nii.gz /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_CHOASYM.nii.gz'); % path
                    % End CHO

                    % Save the interpolated z-spectrum
                    for nd=1:size(cest1_z_b0correct,4)
                        dummy1=squeeze(cest1_z_b0correct(:,:,:,nd));
                        %cest1_z_b0correct_interp2save(:,:,:,nd)=st2(dummy1); % need to transpose so it will save correctly
                        cest1_z_b0correct_interp2save(:,:,:,nd)=(dummy1); % need to transpose so it will save correctly
                        clear dummy1;
                    end
                    save_avw(cest1_z_b0correct_interp2save,strcat(file2save,'OCEST_ZSPECTRUM_', weight_txt),'d',[1.8 2 6 7.692]); % path % this is interpolated at 0.1 ppm and goes -4 ppm to +4 ppm
                    %unix('fslswapdim /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_ZSPECTRUM.nii.gz -x y z /Users/mjd/VU/data/3T/inform2/data/Donahue_TEMPID/DICOM/Donahue_TEMPID_CEST_ZSPECTRUM.nii.gz'); % path
                    % ================================================

                    % end calcualate the APT, NOE (-APT), CHO, B0, Equilibrium, and asymmetry
                    % maps ----------------------------------------------------------------

                    % ======================================================================
                    %
                    % End process the CEST data and save the processed maps
                    %
                    % ======================================================================

                % the lymph offsets just for referene:
                lymphoffsets_noeq=[
                    -5.50
                    -5.10
                    -4.70
                    -4.30
                    -3.90
                    -3.50
                    -3.00
                    -2.60
                    -2.20
                    -1.80
                    -1.40
                    -1.00
                    -0.80
                    -0.60
                    -0.40
                    -0.20
                    0.00
                    0.20
                    0.40
                    0.60
                    0.80
                    1.00
                    1.20
                    1.40
                    1.60
                    1.80
                    2.00
                    2.30
                    2.50
                    2.70
                    2.90
                    3.10
                    3.30
                    3.50
                    3.70
                    3.90
                    4.10
                    4.30
                    4.50
                    4.70
                    4.90
                    5.10
                    5.30
                    5.50];

                %plot(ppm_interp,squeeze(cest1_z_forb0correct(40,40,8,:)),'r.-');
                %figure; plot(lymphoffsets_noeq,squeeze(cest1_z(40,40,8,:)),'b.-');
                end

        clearvars -except preproc_cest mydir outdir myfiles ff good_validated bad_validated val_file processing_problems problem_file weight_types basedirs

    end % end the ff file list loop

    %msg = ['Good:' good_validated '\nBad:' bad_validated];
    %fid = fopen(val_file,'wt');
    %fprintf(fid, msg);
    %fclose(fid);

    %msg = processing_problems;
    %fid = fopen(problem_file,'wt');
    %fprintf(fid, msg);
    %fclose(fid);
end
toc();