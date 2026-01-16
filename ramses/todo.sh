#! /bin/bash

# Argumentos: 
# $1: Tipo de parametrización (periodograma, cepstrum, maxent, mfcc) [default: mfcc]
# $2: Tipo de modelo (Euclidio, Gauss, MixturaGauss, RedNeuronal) [default: Gauss]
# $3+: Parámetros específicos de la parametrización

TIPO_PRM=${1:-mfcc}
TIPO_MOD=${2:-Gauss}

NOM=${TIPO_PRM}_${TIPO_MOD}

DIR_WRK=.

DIR_LOG=$DIR_WRK/Log
FIC_LOG=$DIR_LOG/$(basename $0 .sh).$NOM.log
[ -d $DIR_LOG ] || mkdir -p $DIR_LOG

exec > >(tee $FIC_LOG) 2>&1

hostname
pwd
date 

#Ficheros guia

DIR_GUI=$DIR_WRK/Gui
GUI_ENT=$DIR_GUI/train.gui
GUI_DEV=$DIR_GUI/devel.gui

DIR_SEN=$DIR_WRK/Sen
DIR_MAR=$DIR_WRK/Sen
DIR_PRM=$DIR_WRK/Prm/$NOM
DIR_MOD=$DIR_WRK/Mod/$NOM
FIC_MOD=$DIR_MOD/vocales.mod
DIR_REC=$DIR_WRK/Rec/$NOM

LIS_MOD=$DIR_WRK/Lis/vocales.lis

FIC_RES=$DIR_WRK/Res/$NOM.res
[ -d $(dirname $FIC_RES) ] || mkdir -p $(dirname $FIC_RES)

# Parametrización

dirSen="-s $DIR_SEN"
dirPrm="-p $DIR_PRM"

# Definición de la función de parametrización
FUNK_PRM=$TIPO_PRM

if [ $FUNK_PRM == periodograma ]; then
    EXEC_PREV=$DIR_PRM/$FUNK_PRM
    EPS=${3:-1e-10}

    [ -d $(dirname $EXEC_PREV) ] || mkdir -p $(dirname $EXEC_PREV)
    echo "import numpy as np" | tee $EXEC_PREV
    echo "def $FUNK_PRM(x):" | tee -a $EXEC_PREV
    echo "    return 10*np.log10($EPS+abs(np.fft.fft(x))**2)" | tee -a $EXEC_PREV

elif [ $FUNK_PRM == cepstrum ]; then
    EXEC_PREV=$DIR_PRM/$FUNK_PRM
    EPS=${3:-1e-10}
    NUM_COF=${4:-20}
    [ -d $(dirname $EXEC_PREV) ] || mkdir -p $(dirname $EXEC_PREV)
    echo "import numpy as np" | tee $EXEC_PREV
    echo "def $FUNK_PRM(x):" | tee -a $EXEC_PREV
    echo "    Sx=10*np.log10($EPS+abs(np.fft.fft(x))**2)" | tee -a $EXEC_PREV
    echo "    cepstrum = np.real(np.fft.ifft(Sx))" | tee -a $EXEC_PREV
    echo "    return cepstrum[:$NUM_COF]" | tee -a $EXEC_PREV

elif [ $FUNK_PRM == maxent ]; then
    EXEC_PREV=$DIR_PRM/$FUNK_PRM
    ORDEN_LPC=${3:-12}
    [ -d $(dirname $EXEC_PREV) ] || mkdir -p $(dirname $EXEC_PREV)
    echo "from ramses.maxent import extraer_maxent" | tee $EXEC_PREV
    echo "def $FUNK_PRM(x):" | tee -a $EXEC_PREV
    echo "    return extraer_maxent(x, p=$ORDEN_LPC)" | tee -a $EXEC_PREV

elif [ $FUNK_PRM == mfcc ]; then
    EXEC_PREV=$DIR_PRM/$FUNK_PRM
    NUM_COEF=${3:-13}
    NUM_FILT=${4:-26}
    [ -d $(dirname $EXEC_PREV) ] || mkdir -p $(dirname $EXEC_PREV)
    echo "from ramses.mfcc import extraer_mfcc" | tee $EXEC_PREV
    echo "def $FUNK_PRM(x):" | tee -a $EXEC_PREV
    echo "    return extraer_mfcc(x, numcep=$NUM_COEF, nfilt=$NUM_FILT)" | tee -a $EXEC_PREV

else 
    echo "Parametrización desconocida ($FUNK_PRM)"
    exit 1
fi

funkPrm="-f $FUNK_PRM"
execPrev="-e $EXEC_PREV"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

EXEC="python3 $SCRIPT_DIR/parametriza.py $dirSen $dirPrm $funkPrm $execPrev $GUI_ENT $GUI_DEV"
echo $EXEC && $EXEC || exit 1

# Entrenamiento

dirPrm="-p $DIR_PRM"
dirMar="-m $DIR_MAR"
lisUni="-l $LIS_MOD"
ficMod="-M $FIC_MOD"

if [ "$TIPO_MOD" != "Gauss" ]; then
    clsMod="-C $TIPO_MOD"
else
    clsMod=""
fi

EXEC="python3 $SCRIPT_DIR/entrena.py $dirPrm $dirMar $lisUni $ficMod $clsMod $GUI_ENT"
echo $EXEC && $EXEC || exit 1

# Reconocimiento 

dirRec="-r $DIR_REC"
dirPrm="-p $DIR_PRM"
ficMod="-M $FIC_MOD"

if [ "$TIPO_MOD" != "Gauss" ]; then
    clsMod="-C $TIPO_MOD"
else
    clsMod=""
fi

EXEC="python3 $SCRIPT_DIR/reconoce.py $dirRec $dirPrm $ficMod $clsMod $GUI_DEV"
echo $EXEC && $EXEC || exit 1

# Evaluación del resultado 

dirRec="-r $DIR_REC" 
dirMar="-m $DIR_MAR" 

EXEC="python3 $SCRIPT_DIR/evalua.py $dirRec $dirMar $GUI_DEV"
echo $EXEC && $EXEC | tee $FIC_RES || exit 1

date
echo sacabao, chula
