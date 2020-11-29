param (
    $scriptDirectory,
    $directory
)

# cd $scriptDirectory

# $filenames = Get-ChildItem -Path $directory -Name

# Foreach ($file in $filenames) 
# {
#     python "$scriptDirectory/test.py" $directory\$file
# }

python "combine_datasets.py" "data/objects_detected.jsonl" "data/open_images_test_captions.jsonl"
