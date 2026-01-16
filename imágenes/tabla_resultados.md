# Resultados de los Experimentos

## Comparación por Técnica de Extracción de Características


### CEPSTRUM

| Modelo Acústico | Exactitud |
|-----------------|----------:|
| RedNeuronal | 90.85% |
| Gauss | 87.45% |

### MAXENT

| Modelo Acústico | Exactitud |
|-----------------|----------:|
| RedNeuronal | 79.70% |
| Gauss | 48.95% |
| Euclidio | 45.50% |

### MFCC

| Modelo Acústico | Exactitud |
|-----------------|----------:|
| RedNeuronal | 96.25% |
| Gauss | 91.70% |
| Euclidio | 89.05% |

## Ranking General

| Posición | Sistema | Exactitud |
|----------|---------|----------:|
| 1 🥇 | mfcc + RedNeuronal | **96.25%** |
| 2 🥈 | mfcc + Gauss | **91.70%** |
| 3 🥉 | cepstrum + RedNeuronal | **90.85%** |
| 4  | mfcc + Euclidio | **89.05%** |
| 5  | cepstrum + Gauss | **87.45%** |
| 6  | maxent + RedNeuronal | **79.70%** |
| 7  | maxent + Gauss | **48.95%** |
| 8  | maxent + Euclidio | **45.50%** |
