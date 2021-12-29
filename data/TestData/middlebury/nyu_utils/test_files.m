
homedir = pwd;
folders = strsplit(ls);
folders(strcmp(folders, '') ~= 0) = [];
folders(strcmp(folders, 'test_files.m') ~= 0) = [];
nanfiles = cell(0);
idx = 1;
for ii = 1:length(folders)
    cd(homedir); 
    disp(folders{ii});
    cd(folders{ii});
    try 
        files = strsplit(ls('spad*.mat'));
    catch e
        disp(e.message);
        continue;
    end
    files(strcmp(files, '') ~= 0) = [];
    for jj = 1:length(files)
        load(files{jj});
        if any(isnan(spad(:)))
             fprintf('FOUND SPAD NAN: %s/%s\n',folders{ii}, files{jj});
             nanfiles{idx} = sprintf('%s/%s',folders{ii},files{jj});
             idx = idx + 1;
        end
        if any(isnan(bin(:)))
            fprintf('FOUND BIN NAN: %s/%s\n',folders{ii}, files{jj});
            nanfiles{idx} = sprintf('%s/%s',folders{ii},files{jj});
            idx = idx + 1;
        end
    end
end

disp('Removing nanfiles');
cd(homedir)
for ii = 1:length(nanfiles)
    delete(nanfiles{ii});
    fprintf('Removed %s\n',nanfiles{ii});
end
