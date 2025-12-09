; Makerspace RAG - Inno Setup Installer Script
; Compile with Inno Setup 6.x

#define MyAppName "Makerspace RAG"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Hogskolen i Ostfold"
#define MyAppURL "https://github.com/your-repo/makerspace-rag"
#define MyAppExeName "MakerspaceRAG.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=MakerspaceRAG_Setup_{#MyAppVersion}
SetupIconFile=..\app\static\makerspace-logo.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "norwegian"; MessagesFile: "compiler:Languages\Norwegian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "installollama"; Description: "Install Ollama (required for AI)"; GroupDescription: "Components:"; Flags: checkedonce
Name: "installmariadb"; Description: "Install MariaDB (required for database)"; GroupDescription: "Components:"; Flags: checkedonce

[Files]
; Main application (PyInstaller output)
Source: "..\dist\MakerspaceRAG\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Database setup files
Source: "database_setup.sql"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "setup_database.py"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "setup_ollama.py"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "launcher.py"; DestDir: "{app}\installer"; Flags: ignoreversion

; Environment template
Source: "..\.env.example"; DestDir: "{app}"; DestName: ".env.example"; Flags: ignoreversion

; Ollama installer (bundled)
Source: "OllamaSetup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall skipifsourcedoesntexist; Tasks: installollama

; MariaDB installer (bundled) - download separately
Source: "mariadb-*.msi"; DestDir: "{tmp}"; Flags: deleteafterinstall external skipifsourcedoesntexist; Tasks: installmariadb

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Install Ollama if selected
Filename: "{tmp}\OllamaSetup.exe"; Parameters: "/VERYSILENT /NORESTART"; StatusMsg: "Installing Ollama..."; Tasks: installollama; Flags: waituntilterminated

; Pull required models after Ollama install
Filename: "ollama"; Parameters: "pull llama3"; StatusMsg: "Downloading AI model (llama3)..."; Tasks: installollama; Flags: waituntilterminated runhidden
Filename: "ollama"; Parameters: "pull mxbai-embed-large"; StatusMsg: "Downloading embedding model..."; Tasks: installollama; Flags: waituntilterminated runhidden

; Launch application after install
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\embeddings_cache.json"
Type: filesandordirs; Name: "{app}\*.log"

[Code]
// Check if Ollama is already installed
function IsOllamaInstalled(): Boolean;
var
  ResultCode: Integer;
begin
  Result := Exec('ollama', '--version', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
end;

// Check if MariaDB is already installed
function IsMariaDBInstalled(): Boolean;
begin
  Result := DirExists('C:\Program Files\MariaDB 12.1') or
            DirExists('C:\Program Files\MariaDB 11.0') or
            DirExists('C:\Program Files\MariaDB 10.11') or
            RegKeyExists(HKLM, 'SOFTWARE\MariaDB');
end;

// Update task visibility based on installed components
procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpSelectTasks then
  begin
    // Check Ollama
    if IsOllamaInstalled() then
    begin
      WizardForm.TasksList.Checked[1] := False;
      WizardForm.TasksList.ItemEnabled[1] := False;
    end;

    // Check MariaDB
    if IsMariaDBInstalled() then
    begin
      WizardForm.TasksList.Checked[2] := False;
      WizardForm.TasksList.ItemEnabled[2] := False;
    end;
  end;
end;

// Show message if required components not found and not being installed
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;

  if CurPageID = wpSelectTasks then
  begin
    // Check Ollama
    if (not IsOllamaInstalled()) and (not WizardIsTaskSelected('installollama')) then
    begin
      if MsgBox('Ollama is required for the AI features to work. ' +
                'Do you want to continue without installing Ollama?',
                mbConfirmation, MB_YESNO) = IDNO then
      begin
        Result := False;
        Exit;
      end;
    end;

    // Check MariaDB
    if (not IsMariaDBInstalled()) and (not WizardIsTaskSelected('installmariadb')) then
    begin
      MsgBox('MariaDB is required for the database. ' +
             'The application will prompt you to configure the database on first run. ' +
             'You can use a remote database or install MariaDB manually.',
             mbInformation, MB_OK);
    end;
  end;
end;

// Run database setup after installation
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    // The launcher will handle first-run database setup
    // Just ensure the installer directory exists
  end;
end;
