% Convert .m files to .dat

% Get all .m files in current folder
filesA = dir('*.m');
filesB = dir('../*.m');

% Extract just the names
filesNames = string({filesA.name filesB.name});
filesFolders = string({filesA.folder filesB.folder});

test_script_names = string({filesB.name});
ts_exclude_idx = test_script_names ~= "power_control.m";

exclude = ["convert_to_dat.m" test_script_names(ts_exclude_idx)];


for i = 1:length(filesNames)
    name_str = erase(filesNames(i), ".m");
    if contains(filesNames(i), exclude)
        continue
    else
        copyfile(fullfile(filesFolders(i), filesNames(i)), fullfile("../.dat Files", name_str + ".dat"))
    end
end