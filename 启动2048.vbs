' Launcher for game2048.py - runs it with pythonw (no console window)
Set fso = CreateObject("Scripting.FileSystemObject")
game = fso.GetParentFolderName(WScript.ScriptFullName) & "\game2048.py"
CreateObject("WScript.Shell").Run "pythonw """ & game & """", 0, False
