:<<BATCH
@echo off
setlocal ENABLEDELAYEDEXPANSION

Taskkill /IM "executable.exe" /F  >nul 2>&1

for %%a in (%*) do set "argv[%%a]=1"

IF "%1"=="" (
    echo build and   run in   debug mode: [34mbuild_and_run.bat filename.cpp [0m
    echo build and   run in release mode: [35mbuild_and_run.bat filename.cpp --release[0m
    echo build and   run in    ship mode: [36mbuild_and_run.bat filename.cpp --ship[0m
    echo ---
    echo build and debug in     remedyBG: [37mbuild_and_run.bat filename.cpp --debug[0m
) ELSE (
    IF EXIST "main.obj"       ( del hw.obj         )
    IF EXIST "vc140.pdb"      ( del vc140.pdb      )
    IF EXIST "executable.pdb" ( del executable.pdb )
    IF EXIST "executable.exe" ( del executable.exe )
    IF EXIST "executable.exp" ( del executable.exp )
    IF EXIST "executable.ilk" ( del executable.ilk )
    IF EXIST "executable.lib" ( del executable.lib )

    cls

    set SHIPDEF=
    IF defined argv[--ship] (
        echo [36mbuilding in ship mode[0m
        set SHIPDEF=/DSHIP
        set OPTARG=2
    ) ELSE IF defined argv[--release] (
        echo [35mbuilding in release mode[0m
        set OPTARG=2
    ) ELSE (
        echo [34mbuilding in debug mode[0m
        set OPTARG=d
    )

    cl -O!OPTARG! ^
    /d2FH4- ^
    -W4 -wd4201 -wd4127 ^
    /MDd /nologo -fp:except -GR- -EHa- -FC ^
    /I .\opengl ^
    /EHsc ^
    -Z7 ^
    !SHIPDEF! ^
    %1 ^
    /Feexecutable.exe ^
    /link /NODEFAULTLIB:MSVCRT ^
    /LIBPATH:.\opengl ^
    OpenGL32.lib user32.lib gdi32.lib shell32.lib vcruntime.lib ^
    glfw3.lib
    
    

    call :setESC

    IF EXIST "executable.exe" (
        IF defined argv[--debug] (
            echo [32mdebugging in remedyBG[0m
            call _windows_debug_remedybg.bat
        ) ELSE (
            echo [33mrunning executable[0m
            @echo on
            start executable.exe
            @echo off
        )
    )
)


IF EXIST "main.obj"       ( del main.obj       )
IF EXIST "executable.ilk" ( del executable.ilk )

:setESC
for /F %%a in ('"prompt $E$S & echo on & for %%b in (1) do rem"') do set "ESC=%%a"
@echo off
exit /B


endlocal
BATCH

if [ "$#" -eq 0  ] || ! [ -f "$1" ]; then
    echo "build and run   in   debug mode: [34m./build_and_run.bat main.cpp [0m"
    echo "build and run   in release mode: [35m./build_and_run.bat main.cpp --release[0m"
    echo "build and run   in    ship mode: [36m./build_and_run.bat main.cpp --ship[0m"
    echo "---"
    echo "build and debug in        XCode: [37m./build_and_run.bat main.cpp --debug[0m"
else
    # why is this in 2 passes? i actually don't remember
    if [ -f "executable" ]; then
        rm executable
    fi

    clear

    ARCH=
    OPTARG=
    SHIPDEF=
    if [ "$2" = "--ship" ]; then
        echo "[36mbuilding $1 in ship mode[0m"
        ARCH="-arch arm64 -arch x86_64"
        OPTARG=3
        SHIPDEF=-DSHIP
    elif [ "$2" = "--release" ]; then
        echo "[35mbuilding $1 in release mode[0m"
        OPTARG=3
    else
        echo "[34mbuilding $1 in debug mode[0m"
        OPTARG=0
    fi

    clang++ \
        -ftime-trace \
        -c $1 \
        -std=c++11 \
        -fno-strict-aliasing \
        -ferror-limit=4096 \
        -mmacosx-version-min=11.0 \
        -O$OPTARG \
        $SHIPDEF \
        -g -fstandalone-debug \
        -Wall -Wextra \
        -Wshadow \
        -Werror=vla \
        -Wno-deprecated-declarations \
        -Wno-missing-braces \
        -Wno-missing-field-initializers \
        -Wno-char-subscripts \
        -Wno-write-strings \
        -I./opengl   \
        $ARCH \

    clang++ \
        -g \
        -o executable $(basename $1 .cpp).o \
        -L./opengl   \
        -lglfw3 \
        -framework Cocoa -framework OpenGL -framework IOKit -framework QuartzCore \
        $ARCH \
        # -mmacosx-version-min=13.5 \
        # -Wno-c++11-narrowing \
        # -ftime-report \

    if [ -f "executable" ]; then
        if [ "$2" = "--debug" ]; then
            echo "[32mdebugging in XCode[0m"
            source _mac_debug_xcode.sh
        else
            echo "[33mrunning executable[0m"
            ./executable
        fi
    fi

    rm $(basename $1 .cpp).o
fi

