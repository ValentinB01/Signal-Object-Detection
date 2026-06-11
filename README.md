# Numărarea Obiectelor în Semnale Radio Zgomotoase

**Autor:** Bușe Valentin-Cristian 

## 1. Introducere și Formularea Problemei
Această lucrare abordează problema numărării obiectelor în semnale radio reprezentate sub formă de spectrograme zgomotoase. Fiecare imagine de intrare are o dimensiune aproximativă de 128x55 pixeli și conține între 1 și 5 obiecte (semnale) reprezentate ca linii slabe pe un fundal zgomotos. Sarcina este de a determina numărul lor exact.

Deși formulată ca o clasificare în cinci clase, problema are în esență o natură ordinală (1 < 2 < 3 < 4 < 5). Această proprietate a dictat direcția arhitecturală a proiectului, având în vedere că o eroare de tipul "4 în loc de 5" este semnificativ mai puțin gravă decât una de tipul "1 în loc de 5".

## 2. Preprocesarea și Augmentarea Datelor
* **Preprocesare:** Imaginile sunt citite cu OpenCV, convertite din BGR în RGB, și normalizate folosind media și deviația standard ImageNet.
* **Regula de Aur:** Este critic ca normalizarea aplicată la antrenare să fie identică cu cea de la inferență. Un decalaj aici poate cauza colapsul total al predicțiilor.
* **Augmentare:** Configurația finală utilizează augmentări ușoare: flip orizontal/vertical și o ajustare subtilă de luminozitate/contrast. Experimentele au arătat că augmentările agresive acoperă semnalele slabe și sufocă învățarea (scăzând acuratețea la 58.3%).

## 3. Arhitecturile Testate
Au fost dezvoltate și evaluate două modele fundamentale diferite.

### Model 1: CNN Simplu
* **Structură:** 3 blocuri convoluționale (32, 64, 128 canale) de tip Conv-BN-ReLU, urmate de MaxPooling și Dropout2d, cu un singur cap de clasificare.
* **Diagnostic:** Modelul prezenta confuzii frecvente între clasele vecine și o tendință de subestimare a numărului de semnale.

### Model 2: DualHeadSignalCNN (Modelul Final)
* **Structură:** Extractor cu 4 blocuri reziduale (32, 64, 128, 256 canale). Fiecare bloc folosește o conexiune "skip" (`out = F(x) + x`), permițând trecerea liberă a gradientului și antrenarea în profunzime.
* **Inovație (Două Capete):** Extractorul se împarte în două direcții:
    * *Cap de Clasificare* (256 -> 128 -> 5) pentru predicția finală.
    * *Cap de Regresie* (256 -> 64 -> 1) care acționează ca un "profesor auxiliar", gestionând explicit natura ordinală a problemei.
* **Performanță:** Subestimarea a fost redusă considerabil, iar erorile rămase sunt majoritar "off-by-one" (ex: confuzie între 3 și 4).

## 4. Funcții de Cost și Hiperparametri
Funcția de activare de bază este ReLU. Modelul final utilizează o funcție de cost hibridă: `Loss = CrossEntropy + 0.5 * MSE`. Componenta MSE penalizează pătratul erorilor numerice, introducând semnalul ordinal.

**Configurații Testate:**
| Hiperparametru | Valoare | Rezultat / Observație |
| :--- | :--- | :--- |
| Optimizator | AdamW + OneCycleLR | Stabil cu warmup 8-10% |
| Max Learning Rate | 3e-3 | Colaps total (loss blocat la 1.61) |
| Max Learning Rate | 2.5e-3 | Optim (ales) |
| Pondere Regresie | 0.5 | Subestimare redusă la 123 cazuri (ales) |
| Augmentări | Ușoare | Acuratețe 69.7%+ (ales) |
| Dropout Strat Dens | 0.4 | Optim (ales) |

## 5. Abordări Nereușite Documentate
Documentarea experimentelor eșuate a fost esențială pentru conturarea soluției:
* **Mismatch de Normalizare:** Aplicarea normalizării ImageNet doar la inferență a dus la prăbușirea predicțiilor pe o singură clasă.
* **Learning Rate prea mare:** 3e-3 a cauzat un colaps în faza de warmup.
* **Media Mobilă a Greutăților (EMA):** A eșuat deoarece nu sincroniza statisticile interne din straturile BatchNorm.

## 6. Generarea Submisiei (TTA)
Predicția finală utilizează *Test-Time Augmentation (TTA)*. Modelul procesează imaginea originală și încă 3 variante (flip orizontal, flip vertical, ambele). Probabilitățile rezultate sunt mediate pentru o acuratețe sporită. Logica centrală de inferență:

## 7. Concluzii
Evoluția de la un model CNN simplu la arhitectura reziduală dual-head subliniază importanța metodologiei de diagnostic. Fiecare decizie arhitecturală – de la adaptarea preprocesării la adăugarea "profesorului" auxiliar pentru suport ordinal – a fost validată iterativ pentru a rezolva limitările observate la pașii anteriori.
