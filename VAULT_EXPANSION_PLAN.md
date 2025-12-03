# Vault Utvidelse - Implementasjonsplan

**Opprettet:** 2025-12-03  
**Status:** 🔄 Pågår  
**Mål:** Utvide vault.txt med komplett makerspace-kunnskap

---

## Oversikt

| # | Kategori | Antall emner | Status |
|---|----------|--------------|--------|
| 1 | Programvare-guider | 5 | ✅ Ferdig |
| 2 | Sensorer og moduler | 9 | ✅ Ferdig |
| 3 | Motorer og aktuatorer | 5 | ✅ Ferdig |
| 4 | Displays og output | 6 | ✅ Ferdig |
| 5 | Kommunikasjon og protokoller | 5 | ✅ Ferdig |
| 6 | Vinylkutter | 5 | ✅ Ferdig |
| 7 | Materialkunnskap | 6 | ✅ Ferdig |
| 8 | Vedlikehold | 5 | ✅ Ferdig |
| 9 | Prosjektideer | 4 | ✅ Ferdig |
| 10 | Feilmeldinger og diagnostikk | 4 | ✅ Ferdig |
| 11 | Utstyrsspesifikk info (HiØF) | 4 | ✅ Ferdig |

**Totalt:** 58 emner | **Estimert:** ~1500-2000 linjer

---

## Kategori 1: Programvare-guider

Programvare brukerne trenger for å lage design før produksjon.

- [x] **Inkscape for laserkutting**
  - Vektorgrafikk vs raster
  - Strektykkelser for kutting vs gravering
  - Fargelag for forskjellige innstillinger
  - Eksport til laser (SVG, PDF, DXF)
  - Vanlige feil og løsninger

- [x] **Tinkercad for 3D-modellering**
  - Grunnleggende former og primitiver
  - Gruppering og hull
  - Workplane og alignment
  - Import/eksport STL
  - Tips for printbare design

- [x] **Fusion 360 intro**
  - Sketches og constraints
  - Extrude, revolve, loft
  - Parametrisk design
  - Timeline og redigering
  - Eksport for 3D-printing

- [x] **FreeCAD intro**
  - Part Design workbench
  - Sketcher basics
  - Padding og pockets
  - Open source alternativ til Fusion

- [x] **PrusaSlicer avansert**
  - Variable layer height
  - Modifier meshes
  - Custom supports
  - Seam placement
  - Sequential printing

---

## Kategori 2: Sensorer og moduler

Utvidet sensor-bibliotek for Arduino og Raspberry Pi prosjekter.

- [x] **Ultralyd avstandssensor (HC-SR04)**
  - Hvordan den fungerer
  - Kobling til Arduino
  - Kodeeksempel
  - Begrensninger og tips

- [x] **Temperatursensor (DHT11/DHT22)**
  - Forskjell mellom modellene
  - Bibliotek-installasjon
  - Lesing av temperatur og fuktighet
  - Feilsøking

- [x] **PIR bevegelsessensor**
  - Hvordan den fungerer
  - Justering av sensitivitet og forsinkelse
  - Bruksområder (alarm, automatisk lys)

- [x] **Lyssensor (LDR/fotomotstand)**
  - Voltage divider oppsett
  - Analog avlesning
  - Automatisk lysstyring

- [x] **Jordfuktighet og vannivå**
  - Kapasitiv vs resistiv
  - Plantevanning-prosjekt

- [x] **Trykksensor / Force Sensitive Resistor**
  - Hvordan den virker
  - Kalibrering
  - Bruksområder

- [x] **IR-sensor (hindring, linjefølging)**
  - IR-par (sender/mottaker)
  - Robotnavigasjon

- [x] **Hall-effekt sensor**
  - Magnetfeltdeteksjon
  - RPM-måling

- [x] **Akselerometer/Gyroskop (MPU6050)**
  - 6-akse bevegelsessensor
  - I2C-kommunikasjon
  - Bevegelsesdeteksjon

---

## Kategori 3: Motorer og aktuatorer

Bevegelse og mekanisk kontroll.

- [x] **DC-motorer med L298N H-bro**
  - Hvorfor du trenger H-bro
  - Retningskontroll
  - Hastighetskontroll med PWM
  - Kobling og strømforsyning

- [x] **Steppermotorer (28BYJ-48, NEMA17)**
  - Forskjell stepper vs DC
  - ULN2003 driver (28BYJ-48)
  - A4988/DRV8825 driver (NEMA17)
  - Steg og mikrosteg

- [x] **Servomotorer utvidet**
  - Kontinuerlig vs standard servo
  - Joystick-kontroll
  - Flere servoer samtidig
  - Strømforsyning

- [x] **Releer for 230V**
  - VIKTIG sikkerhetsinformasjon
  - Optocoupler-releer
  - Kobling og isolasjon
  - Når bruke solid-state vs mekanisk

- [x] **Pumper og solenoider**
  - Vannpumper for plantevanning
  - Solenoider for låser
  - Flyback-diode beskyttelse

---

## Kategori 4: Displays og output

Visuell og auditiv feedback.

- [x] **LCD 16x2 med I2C**
  - I2C-adapter fordeler
  - LiquidCrystal_I2C bibliotek
  - Tekst og custom characters
  - Scrolling tekst

- [x] **OLED display (SSD1306)**
  - I2C vs SPI
  - Adafruit GFX bibliotek
  - Grafikk og fonter
  - Animasjoner

- [x] **7-segment display**
  - Felles katode vs anode
  - Multiplexing for flere sifre
  - TM1637 modul (enklere)

- [x] **LED-strips (NeoPixel/WS2812B)**
  - Addresserbare vs vanlige strips
  - FastLED vs Adafruit NeoPixel
  - Animasjoner og effekter
  - Strømberegning

- [x] **Buzzere og piezo**
  - Aktiv vs passiv buzzer
  - Tone()-funksjonen
  - Melodier og alarmer

- [x] **RGB LED**
  - Felles katode vs anode
  - Fargeblanding med PWM
  - Biblioteker for enklere kontroll

---

## Kategori 5: Kommunikasjon og protokoller

Hvordan enheter snakker sammen.

- [x] **I2C protokoll**
  - Master/slave konsept
  - Adressering
  - Koble flere enheter på samme buss
  - Vanlige I2C-moduler

- [x] **SPI protokoll**
  - MOSI, MISO, SCK, CS
  - Når bruke SPI vs I2C
  - Hastighetsfordeler

- [x] **Serial/UART**
  - TX/RX kommunikasjon
  - Baud rate
  - Arduino til Arduino
  - Debugging med Serial Monitor

- [x] **WiFi med ESP8266/ESP32**
  - NodeMCU og Wemos
  - Koble til nettverk
  - Webserver på mikrokontroller
  - IoT-muligheter

- [x] **Bluetooth med HC-05/HC-06**
  - Paring og konfigurasjon
  - Serial over Bluetooth
  - App-kontroll

---

## Kategori 6: Vinylkutter

Komplett guide for vinylkutting.

- [x] **Grunnleggende**
  - Hva er vinylkutting
  - Bruksområder (skilt, t-skjorter, dekaler)
  - Forskjell kutting vs print-og-kutt

- [x] **Materialer**
  - Adhesiv vinyl (permanent vs removable)
  - HTV/transfervinyl for tekstil
  - Sticker-papir
  - Spesialmaterialer (glitter, holografisk)

- [x] **Design for vinylkutter**
  - Vektorgrafikk-krav
  - Inkscape til Silhouette/Cricut
  - Tekst til kurver
  - Mirror for HTV

- [x] **Weeding og overføringstape**
  - Weeding-teknikker
  - Overføringstape-typer
  - Påføring på overflater

- [x] **Feilsøking**
  - Kutter ikke gjennom
  - Løfter materiale
  - Registreringsfeil

---

## Kategori 7: Materialkunnskap

Dybdekunnskap om materialer for alle teknologier.

- [x] **Filamenttyper i dybden**
  - PLA+ vs standard PLA
  - Silk/silky filament
  - Wood-filled
  - Karbonfiber-forsterket
  - Flex/TPU
  - ASA (utendørs-alternativ til ABS)

- [x] **Akryltyper**
  - Støpt vs ekstrudert
  - Farger og transparens
  - Tykkelser og bruksområder
  - Liming og bøying

- [x] **Tretyper for laser**
  - MDF (fordeler/ulemper)
  - Bjørkekryssfiner
  - Balsa og andre myke treslag
  - Behandlet vs ubehandlet

- [x] **Lær og kunstlær**
  - Ekte lær for laser
  - Vegansk/kunstlær (PU-basert OK, PVC NEI)
  - Innstillinger og finish

- [x] **Stoff og tekstil**
  - Hvilke stoffer kan laserkuttes
  - Kanter og fraying
  - Sikkerhet

- [x] **Papir og kartong**
  - Tykkelser
  - Innstillinger for rent kutt
  - Gravering på papir

---

## Kategori 8: Vedlikehold

Holde utstyret i god stand.

- [x] **3D-printer vedlikehold**
  - Dysebytte (når og hvordan)
  - Smøring av akser
  - Beltestramming
  - Rengjøring av byggeplate
  - Sjekkliste for jevnlig vedlikehold

- [x] **Extruder-kalibrering (e-steps)**
  - Hvorfor kalibrere
  - Steg-for-steg måling
  - Lagring i firmware

- [x] **Laserlinse rengjøring**
  - Når rengjøre
  - Riktig rengjøringsmiddel
  - Teknikk for ikke å skade

- [x] **Laser speil-justering**
  - Når det trengs
  - Grunnleggende justering
  - Når kalle service

- [x] **Generell maskinpleie**
  - Støvfjerning
  - Kabelsjekk
  - Firmware-oppdateringer
  - Backup av innstillinger

---

## Kategori 9: Prosjektideer

Inspirasjon for brukere som ikke vet hva de skal lage.

- [x] **Nybegynnerprosjekter per teknologi**
  - 3D-print: Telefonholder, kabelholder, nøkkelring
  - Laser: Navneskilt, coasters, enkel boks
  - Arduino: Nattlys, døralarm, termostat-display
  - Lodding: LED-badge, enkel krets

- [x] **Mellomvanskelige prosjekter**
  - 3D-print: Gir, snap-fit bokser, threads
  - Laser: Living hinge, inlay, lagdelt kunst
  - Arduino: Værsstasjon, automatisk plantevanner
  - Kombinert: Laserkuttet kabinett med elektronikk

- [x] **Kombineringsprosjekter**
  - 3D-print + laser (chassis + paneler)
  - Arduino + 3D-print (sensorhus, robotdeler)
  - Alle teknologier sammen

- [x] **Nyttige hverdagsgjenstander**
  - Veggknagger
  - Skuff-organisering
  - Kabelmanagement
  - Verktøyholdere

---

## Kategori 10: Feilmeldinger og diagnostikk

Spesifikke feilmeldinger og hva de betyr.

- [x] **3D-printer feilmeldinger**
  - MINTEMP / MAXTEMP
  - Thermal runaway
  - Heating failed
  - Probing failed
  - Filament runout
  - Crash detection

- [x] **Arduino feilmeldinger**
  - avrdude: stk500_recv(): programmer is not responding
  - Board not found
  - Compilation errors (vanlige)
  - Out of memory

- [x] **Raspberry Pi feilmeldinger**
  - Kernel panic
  - SD-kort feil
  - Under-voltage warning
  - GPIO-relaterte errors

- [x] **Slicer-advarsler**
  - "Object outside print area"
  - "Supports needed"
  - "Thin walls detected"
  - "Non-manifold edges"

---

## Kategori 11: Utstyrsspesifikk info (HiØF)

Spesifikke instruksjoner for utstyret på labben.

- [x] **Prusa Mini+ spesifikke instruksjoner**
  - Filamentbytte-prosedyre
  - First layer kalibrering
  - SD-kort bruk
  - Vanlige problemer på denne modellen

- [x] **Epilog Fusion M2 40 spesifikke instruksjoner**
  - Oppstartsprosedyre
  - Fokusverktøy bruk
  - Dashboard-innstillinger
  - Vedlikeholdsrutiner

- [x] **Glowforge Pro spesifikke instruksjoner**
  - Cloud-basert workflow
  - Proofgrade materialer
  - Kameraposisjonering
  - Begrensninger

- [x] **Wegstr/Avid CNC spesifikke instruksjoner**
  - Homing-prosedyre
  - Verktøybytte
  - Arbeidsstykke-festing
  - Nødprosedyrer

---

## Fremdrift

Når en kategori er ferdig, oppdater status:
- ⏳ Venter
- 🔄 Pågår  
- ✅ Ferdig

---

## Kommandoer

Si **"fortsett"** eller **"kategori X"** for å starte neste seksjon.

