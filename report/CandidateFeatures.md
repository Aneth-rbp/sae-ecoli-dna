# Características candidatas del SAE

SAE TopK de 128 características con k = 32 (#31), entrenado sobre ventanas genómicas y aplicado a los 3,583 genes. El enriquecimiento usa los 50 genes con mayor activación de cada característica (top-N), con prueba hipergeométrica y corrección BH por característica, después de colapsar las secuencias idénticas (#26). Las cuatro características vienen de `results/CandidateFeatures.tsv`. Dos se presentan como señal funcional y dos como factores de confusión.

Referencia del genoma completo: mediana de longitud del CDS de 900 nt, mediana de GC de 0.524, y 44 genes de 150 nt o menos (1.2 %).

## Característica 26: transposición

- **Términos:** transposition (p ajustada 3.0e-31; 19 de los 23 genes del término están en el top-50), DNA transposition (1.3e-29), transposase activity (1.7e-26).
- **Genes:** insF1-5, insI1-3, insH5, insL1-3, insG, insE1.
- **Interpretación:** responde a las transposasas de las secuencias de inserción (IS). Es la señal más fuerte del análisis y se mantiene después de colapsar las copias idénticas, así que no viene de contar varias veces el mismo gen. Las familias IS tienen una secuencia muy conservada y distinta del resto del genoma; esto explica que una sola característica las agrupe, aunque también significa que el SAE puede estar reconociendo una familia de secuencias más que una función en abstracto.

## Característica 47: lipopolisacárido

- **Términos:** lipopolysaccharide biosynthetic process (8.7e-11; 13 de 58 genes), polysaccharide biosynthetic process (4.2e-08).
- **Genes:** waaS, waaU, waaL, waaZ, rfbX, wbbK, gnsB, nanX, yibA.
- **Interpretación:** agrupa los genes de síntesis del núcleo y del antígeno O del lipopolisacárido (clústeres *waa* y *rfb*). Estos clústeres se adquirieron por transferencia horizontal y tienen un GC bajo: la mediana de GC del top-50 es 0.367, contra 0.524 del genoma. El enriquecimiento es real, pero lo más probable es que la característica codifique composición de bases y que la función aparezca porque estos genes comparten ese origen.

## Característica 118: unión a nucleótidos (factor de confusión: expresión)

- **Términos:** purine ribonucleotide binding (3.6e-17; 32 de 377 genes), ribonucleotide binding (2.1e-16).
- **Genes:** rpoC, groL, rpsA, rho, carB, tufA, tufB, rne, clpX, glnA.
- **Interpretación:** casi todos los genes con mayor activación se cuentan entre los más expresados de la célula (ARN polimerasa, chaperonina GroEL, factor de elongación Tu, proteína ribosomal S1). Los términos de unión a ATP/GTP son muy amplios y abundan en esas proteínas. La hipótesis es que la característica capta el sesgo de codones de los genes muy expresados, no la unión a nucleótidos. No se comprobó: haría falta comparar la activación con un índice de uso de codones (CAI) o con datos de expresión.

## Característica 55: pilus (factor de confusión: longitud)

- **Términos:** pilus (1.0e-13; 13 de 37 genes), cell adhesion (2.4e-11), transcriptional attenuation (9.5e-11; 7 de 9 genes).
- **Genes:** pheM, hisL, thrL, leuL, ivbL, pyrL, mgtL, rhoL, tnaC, ibsA, ibsC, shoB.
- **Interpretación:** 19 de los 50 genes con mayor activación miden 150 nt o menos, contra el 1.2 % del genoma, y la mediana de longitud del top-50 es 472 nt. Los primeros son péptidos líder de atenuación y toxinas pequeñas. Como los embeddings se promedian sobre la secuencia, un gen corto produce un vector distinto solo por su longitud. El término "pilus" probablemente sale de las subunidades fimbriales, que también son más cortas que la mediana del genoma (alrededor de 550 nt). Sirve como advertencia: un término significativo no basta para afirmar que la característica es funcional.

## Excluida

La característica 36 (rango 3; carnitine metabolic process, 8.0e-15) no se interpreta: sus genes con mayor activación (yfcE, ygjR, wcaA, kefF) no tienen relación con el término, y el enriquecimiento depende de pocos genes.
