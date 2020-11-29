param (
    $scriptDirectory,
    $imageDirectory
)

$filenames = Get-ChildItem -Path $imageDirectory -Name

Foreach ($file in $filenames) 
{
    python "$scriptDirectory/test.py" $imageDirectory\$file
}

python "combine_datasets.py" "data/objects_detected.jsonl" "data/open_images_test_captions.jsonl"
