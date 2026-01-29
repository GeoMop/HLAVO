# Data o geometrii geologie v okolí Uhelné

Hlavní popis: `Navod_k_datum.docx` je nedostatečný, popisuje geologii, ale ne strukturu dat 

## Shape file vrstvy
Ohraniceni_modelu_nove.shp                       = větší rozsah modelu se zahrnutím dolu
Rozsah_interpolovanych_oblasti_posttekt.shp      = hlavní geologické dělení
Rozsah_interpolovanych_oblasti.shp               = ekvivalentní s *_syntekt
Rozsah_interpolovanych_oblasti_syntekt.shp       = jemnější dělení
Rozsahy_hornin_pro_interpolace.shp               = patrně geologická mapa - datoé pole "Horizon" indikuje snad který horizont vystupuj na povrch (nesedí pro Turow)
Vrty_z_xlsm_do_interpolace.shp                   = poměrně hustá množina virtuálních vrtů viz. soubor Makro_Vrty_Turow_CZ_PL_upravy_vrtu_podle_modelu.xlsm
Vymezeni_oblasti_NESAT_2023_05_12.shp            = vymezení oblasti modelu nesaturované zóny


## Virtuální  vrty
Jsou patrně ve souboru: `Makro_Vrty_Turow_CZ_PL_upravy_vrtu_podle_modelu.xlsm`

- jedná se o aplikaci ve VB pro tvorbu těch virtuálních vrtů a jejich kontrolu vůči Move 
- chybí popis jednotlivých listů, takže se neorientuju co je vlastně výstup
- není jasná vazba s dalšími datovými soubory



## Final_Gridy_Ondra
- TIFF soubory
- chybějící georeferencování (TFW) nastaveno na `DEM_Turow_10m_resample.tfw` dovnitř souboru TIFF pomocí:
  for f in *.tif *.tiff; do b="${f%.*}"; gdal_translate -a_srs EPSG:5514 "$f" "${b}_georef.tif"; done
- oblasti se překrývají i v rámci stejného horizontu, takže to nemusí být dobře 
- hodilo by se pořadí horizontů (není zřejmé pro negeologa)
- prefixy názvů souborů asi odkazují na geologické vrstvy
  ovšem v rámci prefixu jsou podoblasti, které se opakují jen někde a někde ne.
- Pokud se podaří primárně pracovat s virtuálními vrty, tak bychom to nemuseli potřebovat.

## Popis gridových vrstev pro HG model

Pokud není explicitně v názvu, je každý grid bází příslušné vrstvy, 

- Sx - krystalinikum (top)
  Pw 
- Terciér
  - TCb1
  - Mw
  - TCb2
  - Nw (pokud pod CB3 -> též značeno Nd)
  - TCb3 (může chybět)
  - Ng   (může chybět ?)

- Kvartér (báze a top jílových nepropustných vrstev)
  - Q6 base
  - Q6 top
  ...
  - Q1 Top
  
-  Nd implicitně v Nw
? stejná geo vrstva v souboru -> nepřekrývají se?
- můžu stejnou předponu sloučit do jednoho gridu?

Algoritmus:
pořadí: [Pw, TCb1, Mw, TCb2, Nw, TCb3, Ng, Q6_base, Q6_top, Q5_base, Q5_top, Q3_base, Q3_top, Q2_base, Q2_top, Q1_base, Q1_top]
- všude nastavím výšku podle Sx a typ Pw
- pro každou další vrstvu V
- pokud je face bottom nebo nic:
  - pokud je výška vyšší než poslední výška -> ukončuju předchozí vrstvu a začínám novou vrstvu V na nové výšce
- pokud je top, otestuji  že je výška > než existující a začínám (implicitní) vrstvu Q_štěrky

Celkem vodivostí:
PW
TCb1, 2, 3, (pro začátek stejné)
Mw
Nw, Ng
Q - možno částečně odškálovat podle hloubky
Q_stěrk
min 5 max cca 13
