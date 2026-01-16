#! /usr/bin/env python3

import numpy as np
from tqdm import tqdm

from ramses.util import * 
from ramses.prm import * 
from ramses.mod import *
from ramses.euclidio import Euclidio
from ramses.gaussiano import Gauss
from ramses.mixgauss import MixturaGauss
from ramses.neuronal import RedNeuronal


def reconoce(dirRec, dirPrm, ficMod, *guiSen, ClsMod=Gauss):
    """
    Reconoce la unidad cuyo modelo se ajusta mejor
    """
    modelo = ClsMod(pathMod=ficMod)

    for señal in tqdm(leeLis(*guiSen), ascii="·|/-\\#"):
        pathPrm = pathName(dirPrm, señal, 'prm')
        prm = leePrm(pathPrm)

        reconocida = modelo (prm)

        pathRec = pathName(dirRec, señal, '.rec')
        chkPathName(pathRec)
        with open(pathRec, 'wt') as fpRec: 
            fpRec.write(f'LBO:,,,{reconocida}\n')  

if __name__ == "__main__":
    from docopt import docopt
    import sys

    usage=f"""
Reconoce una base de datos de señales parametrizadas 

usage:
    {sys.argv[0]} [options] <guia> ...
    {sys.argv[0]} -h | --help
    {sys.argv[0]} --version

options:
    -r, --dirRec PATH  Directorio con los ficheros del resultado [default: .]
    -p, --dirPrm PATH  Directorio con las señales parametrizadas [default: .]
    -M, --ficMod PATH  Fichero con el modelo resultante [default: Mod/vocales.mod]
    -e, --execPrev SCRIPT  script de ejecución previa 
    -C, --classMod CLASS  Clase que implementa el modelado acústico
"""
    
    args = docopt(usage, version="tecparla2025")
    dirRec = args["--dirRec"]
    dirPrm = args["--dirPrm"]
    ficMod = args["--ficMod"]
    guiSen = args["<guia>"]
    if args["--execPrev"]: exec(open(args["--execPrev"]).read())
    
    # Mapear nombre de clase a la clase real
    clases_disponibles = {
        'Modelo': Modelo,
        'Euclidio': Euclidio,
        'Gauss': Gauss,
        'MixturaGauss': MixturaGauss,
        'RedNeuronal': RedNeuronal
    }
    
    if args["--classMod"]:
        clsMod = clases_disponibles.get(args["--classMod"], Gauss)
    else:
        clsMod = Gauss
    
    reconoce(dirRec, dirPrm, ficMod, *guiSen, ClsMod=clsMod)



    
