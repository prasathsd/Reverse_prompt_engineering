# PowerShell script to add Gradio frpc to Windows Defender exclusions
$exclusionPath = "C:\Users\Welcome\anaconda3\Lib\site-packages\gradio\frpc_windows_amd64_v0.3"
Add-MpPreference -ExclusionPath $exclusionPath
Write-Host "Added exclusion for Gradio frpc file at: $exclusionPath" 