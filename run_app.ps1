# PowerShell helper to run the Streamlit app using the current Python installation
# Usage: Right-click -> Run with PowerShell or from PowerShell: .\run_app.ps1
Set-Location -Path "$PSScriptRoot\app"
py -m streamlit run main.py $args
