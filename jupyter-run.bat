@echo off
setlocal DisableDelayedExpansion
rem Run from this file's folder, including when opened from another directory.
pushd "%~dp0"
if errorlevel 1 goto directory_failed

if exist "venv\Scripts\python.exe" goto use_venv
if exist "venv" goto broken_venv
goto use_default

:use_venv
echo Using the project virtual environment.
set "PYTHON=venv\Scripts\python.exe"
rem Ensure Jupyter's subcommand lookup also finds the venv's executables.
set "PATH=%CD%\venv\Scripts;%PATH%"
call :ensure_package jupyterlab
if errorlevel 1 goto failed
call :ensure_package ipykernel
if errorlevel 1 goto failed

echo Checking the project Jupyter kernel...
rem Sanitize the folder name and add a path hash to avoid name collisions.
rem Pass paths and display names within Python, never through shell interpolation.
rem Refresh the kernelspec only if its interpreter or display name has changed.
"%PYTHON%" -c "import pathlib,re,hashlib,os,sys,subprocess; from jupyter_client.kernelspec import KernelSpecManager; p=pathlib.Path.cwd(); slug=re.sub('[^a-z0-9._-]+','-',p.name.lower()).strip('-._')[:60] or 'project'; name='venv-'+slug+'-'+hashlib.sha256(os.path.normcase(str(p.resolve())).encode('utf-8')).hexdigest()[:12]; display='Python ('+(p.name or 'Project')+' venv)'; spec=KernelSpecManager().get_all_specs().get(name,{}).get('spec',{}); argv=spec.get('argv',[]); matches=bool(argv) and os.path.normcase(os.path.abspath(argv[0]))==os.path.normcase(os.path.abspath(sys.executable)) and spec.get('display_name')==display; print('Kernel: '+display); sys.exit(0 if matches else subprocess.call([sys.executable,'-m','ipykernel','install','--user','--name',name,'--display-name',display]))"
if errorlevel 1 goto failed

echo Starting JupyterLab...
venv\Scripts\python.exe -m jupyter lab
if errorlevel 1 goto failed
goto done

:use_default
echo No venv found. Using the default py launcher.
set "PYTHON=py"
py -c "import sys; print('Python: '+sys.executable)"
if errorlevel 1 goto failed
call :ensure_package jupyterlab
if errorlevel 1 goto failed
echo Starting JupyterLab...
py -m jupyter lab
if errorlevel 1 goto failed
goto done

:ensure_package
rem Check module availability first; do not upgrade or reinstall on every run.
"%PYTHON%" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('%~1') is not None else 1)"
if not errorlevel 1 exit /b 0
echo Installing missing package: %~1
"%PYTHON%" -m pip --version >nul 2>&1
if not errorlevel 1 goto install_package
"%PYTHON%" -m ensurepip --upgrade
if errorlevel 1 exit /b 1
:install_package
"%PYTHON%" -m pip install %~1
if errorlevel 1 exit /b 1
"%PYTHON%" -c "import %~1"
if errorlevel 1 exit /b 1
exit /b 0

:broken_venv
echo ERROR: A venv exists, but venv\Scripts\python.exe is missing.
echo Repair or recreate the project venv, then run this file again.
goto failed

:failed
echo.
echo ERROR: JupyterLab setup or launch failed. Review the message above.
pause
popd
endlocal
exit /b 1

:directory_failed
echo ERROR: Cannot open the folder containing this batch file.
pause
endlocal
exit /b 1

:done
popd
endlocal
exit /b 0
