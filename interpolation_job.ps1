$directoryPath = "g:\temp\idw\interpolation\source\"
$files = Get-ChildItem -Path $directoryPath -Filter *.txt
foreach ($file in $files) {
    $filePath = $file.FullName
    Start-Job -ScriptBlock {
        param($filePath)
        python "g:\temp\idw\interpolation\interpolation.py" $filePath
    } -ArgumentList $filePath
}

Get-Job | Wait-Job
Get-Job | Receive-Job
