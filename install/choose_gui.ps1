# Window that asks which interface to start. Prints one word on stdout:
#   tk   - desktop GUI (Tkinter)     (Yes)
#   web  - web GUI in the browser    (No)
#   quit - nothing                   (Cancel)
Add-Type -AssemblyName System.Windows.Forms
$text = "Which interface should FIDWAC v2 start?`n`nYes = Desktop GUI (Tkinter)`nNo = Web GUI (browser)`nCancel = quit"
$answer = [System.Windows.Forms.MessageBox]::Show($text, "FIDWAC v2",
    [System.Windows.Forms.MessageBoxButtons]::YesNoCancel,
    [System.Windows.Forms.MessageBoxIcon]::Question)
switch ($answer) {
    "Yes" { Write-Output "tk" }
    "No"  { Write-Output "web" }
    default { Write-Output "quit" }
}
