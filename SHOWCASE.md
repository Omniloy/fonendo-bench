# fonendo-bench showcase

_OmniScribe 2 vs commercial APIs and open-source models · Spanish clinical audio · 15 Sep 2026_

This is the text version of the [showcase page](https://omniloy.github.io/fonendo-bench/) ([source](docs/index.html)). The page has the same numbers and transcripts, with the audio playable inline; here each clip links to its audio file.

## OmniScribe 2 gets more medical terms right than six widely used speech systems, commercial and open source, in their default configuration.

_OmniScribe 2 · Omniloy’s self-hosted clinical transcription_

OmniScribe 2 is Omniloy's self-hosted transcription system, and it uses context from the patient's record (in this test that context included the medical terms spoken in each clip: a best case). On 300 Spanish clinical clips with synthetic voices it gets **85.7%** of medical terms right. The six other systems, three commercial APIs and three open-source models in their default setup, get **50–71%**. On clean audio it also has the **lowest word error rate** of all seven. On degraded audio (noise, phone codecs, reverb), Soniox has a lower word error rate and Cohere Transcribe is level. OmniScribe 2 can also write a look-alike drug name instead of the one spoken.

- **85.7% of medical terms right** across 300 clinical clips. The best of the other six, Soniox, gets 70.8%; the rest get 50–64%. Ahead on clean (92.7% vs ≤77.6%) and on degraded audio (78.6% vs ≤65.1%).
- **3.65% WER on clean clinical audio**, the lowest of the seven (next: Soniox and Whisper-turbo, 4.83%). On real human speech: 6.6%, the lowest average of the seven, level with Cohere Transcribe (6.8%).
- **13.3% WER on degraded audio**, against 9.5% for Soniox (Cohere 13.4%, level). That gives Soniox the lower overall WER (7.18% vs 8.52%).

## The scoreboard

_300 clinical clips · clean and degraded_

Each system is compared with OmniScribe 2 using a paired bootstrap on the same clips. **ours better** and **theirs better** mean the 95% interval of the difference excludes zero; **level** means it includes zero. _Terms right_ counts a term only if every word of it is correct.

**Medical terms right**

| System | Clean | Degraded | All 300 |
|---|---:|---:|---:|
| **OmniScribe 2** (Omniloy, self-hosted, not publicly available) | 92.7% | 78.6% | 85.7% |
| _Commercial APIs, default configuration_ | | | |
| Soniox stt-rt-v5 · API | 76.6% (ours better) | 65.1% (ours better) | 70.8% (ours better) |
| Deepgram Nova-3 · API | 76.0% (ours better) | 49.5% (ours better) | 62.8% (ours better) |
| Deepgram Flux · API | 64.1% (ours better) | 35.4% (ours better) | 49.7% (ours better) |
| _Open-source models, default configuration_ | | | |
| Whisper large-v3-turbo · open | 77.6% (ours better) | 50.0% (ours better) | 63.8% (ours better) |
| Voxtral Mini 4B Realtime · open | 72.4% (ours better) | 47.4% (ours better) | 59.9% (ours better) |
| Cohere Transcribe · open | 71.4% (ours better) | 49.5% (ours better) | 60.4% (ours better) |

**Word error rate**

| System | Clean | Degraded | All 300 |
|---|---:|---:|---:|
| **OmniScribe 2** (Omniloy, self-hosted, not publicly available) | 3.7% | 13.3% | 8.5% |
| _Commercial APIs, default configuration_ | | | |
| Soniox stt-rt-v5 · API | 4.8% (ours better) | 9.5% (theirs better) | 7.2% (theirs better) |
| Deepgram Nova-3 · API | 5.6% (ours better) | 17.2% (ours better) | 11.5% (ours better) |
| Deepgram Flux · API | 7.3% (ours better) | 27.9% (ours better) | 17.7% (ours better) |
| _Open-source models, default configuration_ | | | |
| Whisper large-v3-turbo · open | 4.8% (ours better) | 15.6% (ours better) | 10.3% (ours better) |
| Voxtral Mini 4B Realtime · open | 5.8% (ours better) | 21.8% (ours better) | 13.9% (ours better) |
| Cohere Transcribe · open | 5.2% (ours better) | 13.4% (level) | 9.3% (level) |

OmniScribe 2 used context from the patient's record; in this test that context included the medical terms spoken in each clip: a best case. Clean: 147 clips. Degraded: 153 clips with noise at 20 to 5 dB SNR, and on many clips also a phone or low-bitrate codec, room reverb, or a speed or pitch change. Every other system ran in its default configuration: no custom vocabulary or prompt, and Spanish set as the language wherever the system allows it (Voxtral Realtime has no language setting). Empty outputs count as errors: Flux returned no text on 9 clinical clips, Voxtral Realtime on 7 and Nova-3 on 1. Confidence intervals resample whole sentences, because the same sentence appears in several clips.

## Ahead on medical terms in both conditions; the WER gap is degraded audio

_Where the difference comes from_

Each line runs from the 147 clean clips to the 153 degraded ones. OmniScribe 2 stays ahead of every system on medical terms in both conditions. On word error rate it is the lowest of the seven on clean audio. On degraded audio Soniox is lower and Cohere Transcribe is level, and that is where Soniox’s overall WER lead comes from.

The page draws this as two slope charts, medical terms right and word error rate, from clean to degraded; their values are the Clean and Degraded columns of the scoreboard above. The clean and degraded groups are mostly different sentences, so each line compares two groups of clips, not the same clips getting worse.

## OmniScribe 2 gets the term; the other six miss it

_Listen · medical terms_

In 38 of the 300 clips, OmniScribe 2 gets every medical term right and all six other systems miss at least one; the reverse happens on 3. Against Soniox alone, OmniScribe 2 gets every term where Soniox misses one on 66 clips, and the reverse happens on 19.

Legend: **term** medical term heard correctly · ~~wrong~~ medical term misheard · These cards mark medical terms only. In _What was said_, the medical terms are in bold.

#### `urg_019__kokoro`

clean · synthetic voice · Kokoro · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/urg_019__kokoro.mp3) · [docs/audio/urg_019__kokoro.mp3](docs/audio/urg_019__kokoro.mp3)

**What was said:** Recuerdo al residente que la **cilastatina** no tiene actividad antibacteriana y solo evita la degradación renal del antibiótico asociado.

| System | Transcript |
|---|---|
| OmniScribe 2 | Recuerdo al residente que la **cilastatina** no tiene actividad antibacteriana y solo evita la degradación renal del antibiótico asociado. |
| Soniox stt-rt-v5 | Recuerdo al residente que la ~~silastatina~~ no tiene actividad antibacteriana y solo evita la degradación renal del antibiótico asociado. |
| Deepgram Nova-3 | Recuerdo al residente que la ~~silastatina~~ no tiene actividad antibacteriana, y solo evita la degradación renal del antibiótico asociado. |
| Deepgram Flux | Recuerdo al residente que la ~~silastatina~~ no tiene actividad antibacteriana y solo evita la degradación renal del antibiótico asociado. |
| Whisper large-v3-turbo | Recuerdo al residente que la ~~silastatina~~ no tiene actividad antibacteriana y solo evita la degradación renal del antibiótico asociado. |
| Voxtral Mini 4B Realtime | Recuerdo al residente que la ~~silastatina~~ no tiene actividad antibacteriana y solo evita la degradación renal del antibiótico asociado. |
| Cohere Transcribe | Recuerdo al residente que la ~~silastatina~~ no tiene actividad antibacteriana y solo evita la degradación renal del antibiótico asociado. |

#### `urg_005`

clean · synthetic voice · ElevenLabs · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/urg_005.mp3) · [docs/audio/urg_005.mp3](docs/audio/urg_005.mp3)

**What was said:** Si la crisis migrañosa no cede, valorad **lasmiditan** como alternativa, pero advertid a la paciente de que no conduzca después.

| System | Transcript |
|---|---|
| OmniScribe 2 | Si la crisis migrañosa no cede, valorad **lasmiditan** como alternativa, pero advertid a la paciente de que no conduzca después. |
| Soniox stt-rt-v5 | Si la crisis migrañosa no cede, Valorat ~~las miditan~~ como alternativa, pero advertida la paciente de que no conduzca después. |
| Deepgram Nova-3 | Si la crisis migrañosa no cede, valorad ~~las MIDITAN~~ como alternativa. Pero advertid a la paciente de que no conduzca después. |
| Deepgram Flux | Si la crisis migrañosa no cede, valorad ~~las meditan~~ como alternativa, pero advertid a la paciente de que no conduzca después. |
| Whisper large-v3-turbo | Si la crisis migrañosa no cede, Valorad ~~las miditan~~ como alternativa, pero advertid a la paciente de que no conduzca después. |
| Voxtral Mini 4B Realtime | Si la crisis migrañosa no cede, valorad ~~las miditan~~ como alternativa, pero advertid a la paciente de que no conduzca después. |
| Cohere Transcribe | Si la crisis migrañosa no cede, valorad ~~las miditan~~ como alternativa, pero advertid a la paciente de que no conduzca después. |

#### `prue_054__kokoro`

clean · synthetic voice · Kokoro · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/prue_054__kokoro.mp3) · [docs/audio/prue_054__kokoro.mp3](docs/audio/prue_054__kokoro.mp3)

**What was said:** **Uro-TAC** con **Iomeron** en fase excretora, que el cólico se le repite y el sedimento tiene microhematuria.

| System | Transcript |
|---|---|
| OmniScribe 2 | **Uro-TAC** con **iomeron** en fase excretora, que el cólico se le repite y el sedimento tiene microhematuria. |
| Soniox stt-rt-v5 | Uro ~~te hace~~ con ~~yomeron~~ en fase excretora, que el cólico se le repite y el sedimento tiene microhematuria. |
| Deepgram Nova-3 | ~~Urotheast~~ con en fase excretora, que el cólico se le repite y el sedimento tiene microhematuria.<br>_drops “iomeron”_ |
| Deepgram Flux | ~~Urotease~~ con en fase excretora, que el cólico se le repite y el sedimento tiene microhematuria.<br>_drops “iomeron”_ |
| Whisper large-v3-turbo | ~~Uroteace~~ con **iomeron** en fase excretora, que el cólico se le repite y el sedimento tiene microhematuria. |
| Voxtral Mini 4B Realtime | _(no output)_ |
| Cohere Transcribe | ~~Urotease~~ con **iomeron** en fase excretora, que el cólico se le repite y el sedimento tiene microhematuria. |

#### `prue_064`

clean · synthetic voice · ElevenLabs · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/prue_064.mp3) · [docs/audio/prue_064.mp3](docs/audio/prue_064.mp3)

**What was said:** Cursa un **SPECT cerebral de perfusión** con **Ceretec** dentro del estudio prequirúrgico de la epilepsia refractaria.

| System | Transcript |
|---|---|
| OmniScribe 2 | Cursa un **SPECT cerebral de perfusión** con **ceretec** dentro del estudio prequirúrgico de la epilepsia refractaria. |
| Soniox stt-rt-v5 | Cursa un ~~espectro~~ cerebral de perfusión con **CERETEC** dentro del estudio prequirúrgico de la epilepsia refractaria. |
| Deepgram Nova-3 | Cursa un ~~spec~~ cerebral de perfusión con ~~Zeretec~~, dentro del estudio prequirúrgico de la epilepsia refractaria. |
| Deepgram Flux | Cursa un ~~espén~~ cerebral de perfusión con ~~Cretec~~ dentro del estudio prequirúrgico de la epilepsia refractaria. |
| Whisper large-v3-turbo | Cursa un ~~SPEC~~ cerebral de perfusión con **Ceretec** dentro del estudio prequirúrgico de la epilepsia refractaria. |
| Voxtral Mini 4B Realtime | Cursa un ~~espectro~~ cerebral de perfusión con **Ceretec** dentro del estudio prequirúrgico de la epilepsia refractaria. |
| Cohere Transcribe | Cursa un ~~SPEC~~ cerebral de perfusión con **CERETEC** dentro del estudio prequirúrgico de la epilepsia refractaria. |

#### `mint_059__kokoro`

clean · synthetic voice · Kokoro · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/mint_059__kokoro.mp3) · [docs/audio/mint_059__kokoro.mp3](docs/audio/mint_059__kokoro.mp3)

**What was said:** Se administra **tropisetron** intravenoso por vómitos incoercibles en el contexto del **síndrome de gastroparesia.**

| System | Transcript |
|---|---|
| OmniScribe 2 | Se administra **tropisetron** intravenoso por vómitos incoercibles en el contexto del **síndrome de gastroparesia**. |
| Soniox stt-rt-v5 | Se administra ~~Tropicetron~~ intravenoso por vómitos incoercibles, en el contexto del **síndrome de gastroparesia**. |
| Deepgram Nova-3 | Se administra ~~Tropiceton~~ intravenoso por vómitos incoercibles en el contexto del **síndrome de gastroparesia**. |
| Deepgram Flux | Se administra ~~tropicetron~~ intravenoso por vómitos incuercibles en el contexto del **síndrome de gastroparesia**. |
| Whisper large-v3-turbo | Se administra ~~tropicétron~~ intravenoso por vómitos incoercibles en el contexto del **síndrome de gastroparesia**. |
| Voxtral Mini 4B Realtime | Se administra ~~tropicetron~~ intravenoso por vómitos incoercibles en el contexto del **síndrome de gastroparesia**. |
| Cohere Transcribe | Se administra ~~tropicetrón~~ intravenoso por vómitos incoercibles en el contexto del **síndrome de gastroparesia**. |

#### `mint_036__kokoro__snr10`

SNR 10 dB · 6-voice babble · synthetic voice · Kokoro · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/mint_036__kokoro__snr10.mp3) · [docs/audio/mint_036__kokoro__snr10.mp3](docs/audio/mint_036__kokoro__snr10.mp3)

**What was said:** Se confirma **neumonía micobacteriana** y se inicia **pirazinamida** dentro de la pauta cuádruple durante los dos primeros meses.

| System | Transcript |
|---|---|
| OmniScribe 2 | Se confirma **neumonía micobacteriana** y se inicia **pirazinamida** dentro de la pauta cuádruple durante los dos primeros meses. |
| Soniox stt-rt-v5 | Se confirma **neumonía micobacteriana** y se inicia ~~piracinamida~~ dentro de la pauta cuádruple durante los 2 primeros meses. |
| Deepgram Nova-3 | Se confirma **neumonía micobacteriana** y se inicia ~~piracinamida~~ dentro de la pauta cuádruple durante los 2 primeros meses. |
| Deepgram Flux | se confirma neumonía ~~mycobacterium~~ y se inicia ~~piracinamida~~ dentro de las pautas cuádruple durante los dos primeros meses. |
| Whisper large-v3-turbo | se confirma **neumonía micobacteriana** y se inicia ~~piracinamida~~ dentro de la pauta cuádruple durante los dos primeros meses. |
| Voxtral Mini 4B Realtime | Se confirma **neumonía micobacteriana** y se inicia ~~piracinamida~~ dentro de la pauta cuádruple durante los dos primeros meses. |
| Cohere Transcribe | Se confirma neumonía ~~mycobacteriana~~ y se inicia ~~piracinamida~~ dentro de la pauta cuádruple durante los dos primeros meses. |

## On real speech, it is among the most accurate

_Real speech · three public sets_

On three real-speech sets together (FLEURS, VoxPopuli and MediaSpeech: real speakers, general Spanish, no medical terms), OmniScribe 2 has 6.6% average WER. That is level with Cohere Transcribe (6.8%), which is better on VoxPopuli and worse on MediaSpeech, and significantly ahead of the other five systems on this page (7.4–9.4%), by a paired bootstrap on the three-set average. In the [full leaderboard](results/leaderboard.md), two open models that are not on this page, omniASR-LLM-7B (6.5%) and Voxtral Small 24B (6.6%), have a slightly lower or the same average.

## Where it loses: degraded audio, and look-alike drug names

_Listen · where it loses_

On degraded clips OmniScribe 2 makes 3 or more errors on 47 clips, Soniox on 37. The cases below are clips where Soniox makes at most 2 errors and OmniScribe 2 at least 4 more, plus one clean clip. The recognizer can drop the rest of a sentence or replace it with fluent Spanish that was never said. The clinically serious failure is different: it can write a look-alike drug name instead of the one spoken. This happened on one clean clip too. The next things to fix are noise-robust input, a guard that catches cut-off or invented output, and tighter handling of look-alike drug names.

Legend: **term** correct term · ~~wrong~~ misheard or inserted word · Missing words are listed under each transcript.

#### `mint_014__kokoro`

**Writes a look-alike drug** · clean · synthetic voice · Kokoro · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/mint_014__kokoro.mp3) · [docs/audio/mint_014__kokoro.mp3](docs/audio/mint_014__kokoro.mp3)

**What was said:** Ante la crisis adrenérgica con cifras tensionales extremas se administró **fentolamina** intravenosa antes de introducir el betabloqueo.

| System | Transcript |
|---|---|
| OmniScribe 2 | Ante la crisis adrenérgica con cifras tensionales extremas, se administró ~~fenfluramina~~ intravenosa antes de introducir el ~~beta bloqueo~~.<br>_writes the look-alike drug fenfluramina, on clean audio_ |
| Soniox stt-rt-v5 | Ante la crisis adrenérgica con cifras tensionales extremas, se administró **fentolamina** intravenosa antes de introducir el betabloqueo. |
| Deepgram Nova-3 | Ante la crisis adrenérgica, con cifras tensionales extremas, se administró **fentolamina** intravenosa antes de introducir el betabloqueo. |
| Deepgram Flux | ante la crisis adrenérgica con cifras tensionales extremas, se administró **fentolamina** intravenosa antes de introducir el betabloqueo. |
| Whisper large-v3-turbo | Ante la crisis adrenérgica con cifras tensionales extremas, se administró **fentolamina** intravenosa antes de introducir el ~~beta bloqueo~~. |
| Voxtral Mini 4B Realtime | Ante la crisis adrenérgica con cifras tensionales extremas, se administró **fentolamina** intravenosa antes de introducir el ~~beta bloqueo~~. |
| Cohere Transcribe | Ante la crisis adrenérgica con cifras tensionales extremas, se administró **fentolamina** intravenosa antes de introducir el betabloqueo. |

#### `mint_012__snr20`

**Writes a look-alike drug** · SNR 20 dB · brown noise · pitch -1.2 st · synthetic voice · ElevenLabs · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/mint_012__snr20.mp3) · [docs/audio/mint_012__snr20.mp3](docs/audio/mint_012__snr20.mp3)

**What was said:** Se prescribe **prazosina** como alfabloqueante coadyuvante por mal control tensional refractario, avisando del riesgo de hipotensión de primera dosis.

| System | Transcript |
|---|---|
| OmniScribe 2 | Se prescribe ~~terazosina~~ como ~~el fábricoqueante cuatrovante para el~~ control tensional refractario, avisando del riesgo de hipotensión de primera dosis.<br>_writes the look-alike drug terazosina_ |
| Soniox stt-rt-v5 | Se prescribe **prazosina** como alfabloqueante coadyuvante por mal control tensional refractario; avisando del riesgo de hipotensión de primera dosis. |
| Deepgram Nova-3 | Se prescribe como alfabloqueante ~~o adyuvante~~ por mal control tensional refractario. Avisando del riesgo de hipotensión de ~~1º~~ dosis.<br>_drops “prazosina”_ |
| Deepgram Flux | Se prescribe ~~la cocina~~, como ~~el faloqueante cuadrioante~~, por mal control tensional refractario, avisando del riesgo de hipotensión de primera dosis. |
| Whisper large-v3-turbo | Se prescribe ~~prafusina~~ como alfabloqueante ~~o adyuvante~~ por mal control tensional refractario, avisando del riesgo de hipotensión de primera dosis. |
| Voxtral Mini 4B Realtime | Se prescribe ~~patrocina~~ como alfabloqueante ~~o adyuvante~~ por mal control tensional refractario. Avisando del riesgo de hipotensión de primera dosis. |
| Cohere Transcribe | Se prescribe **prazosina** como alfabloqueante coadyuvante por mal control tensional refractario, avisando del riesgo de hipotensión de primera dosis. |

#### `var_051__kokoro__snr10`

**Stops early** · SNR 10 dB · 6-voice babble · pitch -0.5 st · synthetic voice · Kokoro · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/var_051__kokoro__snr10.mp3) · [docs/audio/var_051__kokoro__snr10.mp3](docs/audio/var_051__kokoro__snr10.mp3)

**What was said:** Mírale **la gaso** de ingreso, que tenía acidosis; en la gasometría arterial de control el pH ya se ha corregido.

| System | Transcript |
|---|---|
| OmniScribe 2 | Mírale **la gaso** de ingreso.<br>_15 words missing (“que tenía acidosis en …”)_ |
| Soniox stt-rt-v5 | Mírale **la gaso** de ingreso, que tenía ~~hace 12~~. En la gasometría arterial de control, el pH ya se ha corregido. |
| Deepgram Nova-3 | ~~Mira el lagazo~~ de ingreso. La gasometría arterial de control ~~de gas y gas~~ se ha ~~corrido~~.<br>_drops “que tenía acidosis en”_ |
| Deepgram Flux | Mírale ~~el agaso~~ de ingreso. La ~~geometría~~ arterial de control ~~en casi~~ ya ~~sea continua~~.<br>_drops “que tenía acidosis en”_<br>_drops “corregido”_ |
| Whisper large-v3-turbo | Mírale ~~el agazo~~ de ingreso que tenía ~~asidosis~~. En la gasometría arterial de control, el ~~gachi~~ ya se ha ~~cogido~~. |
| Voxtral Mini 4B Realtime | Mírale la ~~gasa~~ de ingreso.<br>_15 words missing (“que tenía acidosis en …”)_ |
| Cohere Transcribe | Mírale **la gaso** de ingreso, que tenía acidosis. En la gasometría arterial de control, el pH ya se ~~acortiva~~.<br>_drops “corregido”_ |

#### `var_035__snr15`

**Rewrites the sentence** · SNR 15 dB · 6-voice babble · Opus 16 kbps · 1.08× speed · pitch -0.5 st · synthetic voice · ElevenLabs · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/var_035__snr15.mp3) · [docs/audio/var_035__snr15.mp3](docs/audio/var_035__snr15.mp3)

**What was said:** Le hicieron **una colono** el año pasado, le quitaron tres pólipos y quedó pendiente el resultado.

| System | Transcript |
|---|---|
| OmniScribe 2 | Le hicieron **una colono** el año pasado. ~~Liberaron~~ tres ~~bolígrafos~~ y ~~quedo~~ pendiente ~~los soldados~~.<br>_drops “quitaron”_ |
| Soniox stt-rt-v5 | Le hicieron **una colono** el año pasado. Le quitaron 3 pólipos y quedó pendiente el resultado. |
| Deepgram Nova-3 | Le hicieron **una colono** el año pasado. Le quitaron 3 pólipos y quedó pendiente el resultado. |
| Deepgram Flux | hicieron **una colono** el año pasado. ~~Vidal~~, ~~después cosquilleo~~ pendiente ~~de los soldados~~.<br>_drops “Le”_<br>_drops “pólipos y quedó”_ |
| Whisper large-v3-turbo | Le hicieron **una colono** el año pasado. Le ~~invitaron~~ tres ~~colipos~~ y quedó pendiente el resultado. |
| Voxtral Mini 4B Realtime | Le hicieron **una colono** el año pasado.<br>_9 words missing (“le quitaron tres pólipos …”)_ |
| Cohere Transcribe | Le hicieron **una colono** el año pasado. Le quitaron tres ~~polivos~~ y quedó pendiente ~~de los soldados~~. |

#### `var_048__snr15`

**Rewrites the sentence** · SNR 15 dB · 6-voice babble · reverb · AMR-NB phone 5.9 kbps · 0.92× speed · pitch +0.5 st · synthetic voice · ElevenLabs · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/var_048__snr15.mp3) · [docs/audio/var_048__snr15.mp3](docs/audio/var_048__snr15.mp3)

**What was said:** La **PCR** de ayer estaba en 180; hoy la proteína C reactiva ha bajado a 95, así que el antibiótico funciona.

| System | Transcript |
|---|---|
| OmniScribe 2 | ~~En~~ la **PCR** de ayer estaba en ~~100%~~. Hoy la proteína C reactiva ha bajado a 95. ~~Las fibras~~ antibiótico ~~ocurren~~.<br>_drops “el”_ |
| Soniox stt-rt-v5 | La **PCR** de ayer estaba en 180, hoy la proteína C reactiva ha bajado a 95, así que el antibiótico funciona. |
| Deepgram Nova-3 | La **PCR** de ayer estaba en ~~130~~. Hoy la proteína c reactiva ha bajado a 95. ~~Antibiotico~~ funciona<br>_drops “que el antibiótico”_ |
| Deepgram Flux | _(no output)_ |
| Whisper large-v3-turbo | La **PCR** de ~~miel~~ estaba en 180. Hoy la proteína C reactiva ha bajado a 95. Así que el antibiótico funciona. |
| Voxtral Mini 4B Realtime | La ~~DTR~~ de ayer estaba ~~enciendo central~~. Hoy la proteína C reactiva ha bajado a 95. ~~Las bacterias antibióticas funcionan~~.<br>_drops “funciona”_ |
| Cohere Transcribe | La **PCR** de ayer estaba en ~~160~~. Hoy la proteína C reactiva ha bajado a 95. Así que el antibiótico funciona. |

#### `resu_020__kokoro__snr10`

**Mishears terms** · SNR 10 dB · recorded noise · pitch -1.2 st · synthetic voice · Kokoro · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/resu_020__kokoro__snr10.mp3) · [docs/audio/resu_020__kokoro__snr10.mp3](docs/audio/resu_020__kokoro__snr10.mp3)

**What was said:** En la biopsia prostática sale un **adenocarcinoma de próstata** Gleason 3+4, con dos cilindros de doce afectados.

| System | Transcript |
|---|---|
| OmniScribe 2 | En la biopsia prostática sale ~~una de lo carcinoma~~ de próstata ~~que son~~ tres más cuatro, con dos cilindros de doce afectados. |
| Soniox stt-rt-v5 | En la biopsia prostática sale ~~una~~ **adenocarcinoma de próstata** Gleason 3+4, con 2 cilindros de 12 afectados. |
| Deepgram Nova-3 | En la biopsia prostática, ~~salen la dedrocarcinoa~~ de próstata 3 ~~más 4.2~~ cilindros de 12 afectados.<br>_drops “Gleason”_<br>_drops “dos”_ |
| Deepgram Flux | En la biopsia ~~bastante~~, sale ~~una deucacionada~~ de próstata ~~ya son~~ tres más cuatro, con dos cilindros de doce afectados. |
| Whisper large-v3-turbo | En la biopsia prostática, sale ~~una~~ **adenocarcinoma de próstata** ~~que ya son~~ 3 más 4, con dos cilindros de 12 afectados. |
| Voxtral Mini 4B Realtime | En la biopsia prostática ~~salen~~ **adenocarcinoma de próstata** ~~que son~~ 3 más 4, con dos cilindros de 12 afectados.<br>_drops “un”_ |
| Cohere Transcribe | En la biopsia prostática sale ~~una delocarcinoma~~ de próstata ~~gliason~~ 3 más 4, con dos cilindros de doce afectados. |

## Inserted words: the others do it too

_A fair check · inserted words_

An inserted word is one that appears in the transcript but was never said. On the clinical set, OmniScribe 2’s rate of inserted words is level with Soniox, Nova-3, Voxtral Realtime and Cohere, and lower than Flux and Whisper-turbo. On FLEURS real speech it has the lowest rate of the seven (level with Whisper-turbo and Voxtral Realtime). Its distinctive weakness is the one shown above: occasionally writing a look-alike drug name.

**Inserted words per 1,000**

| System | Clinical · clean | Clinical · degraded | Clinical · all 300 | FLEURS real speech |
|---|---:|---:|---:|---:|
| **OmniScribe 2** (Omniloy, self-hosted, not publicly available) | 5.7 | 14.7 | 10.3 | 2.6 |
| _Commercial APIs, default configuration_ | | | | |
| Soniox stt-rt-v5 · API | 7.9 (level) | 11.2 (level) | 9.6 (level) | 4.3 (ours fewer) |
| Deepgram Nova-3 · API | 9.7 (level) | 18.9 (level) | 14.3 (level) | 9.6 (ours fewer) |
| Deepgram Flux · API | 11.5 (ours fewer) | 25.9 (ours fewer) | 18.7 (ours fewer) | 5.6 (ours fewer) |
| _Open-source models, default configuration_ | | | | |
| Whisper large-v3-turbo · open | 6.1 (level) | 24.5 (ours fewer) | 15.4 (ours fewer) | 3.2 (level) |
| Voxtral Mini 4B Realtime · open | 7.9 (level) | 19.6 (level) | 13.8 (level) | 3.8 (level) |
| Cohere Transcribe · open | 7.9 (level) | 16.1 (level) | 12.0 (level) | 4.6 (ours fewer) |

Per 1,000 reference words, scored the same way for every system; labels use a paired bootstrap against OmniScribe 2. VoxPopuli and MediaSpeech are left out because their references skip some real speech, for example a Greek phrase at the start of one MediaSpeech clip, and any system that transcribes it is counted as inserting words.

#### `var_022__snr5`

**Soniox invents a phrase** · SNR 5 dB · recorded noise · reverb · pitch +1.2 st · synthetic voice · ElevenLabs · others in default setup

[▶ Listen](https://omniloy.github.io/fonendo-bench/audio/var_022__snr5.mp3) · [docs/audio/var_022__snr5.mp3](docs/audio/var_022__snr5.mp3)

**What was said:** Pásate por **la eco** a las diez, que la radióloga te lo hace entre paciente y paciente.

| System | Transcript |
|---|---|
| OmniScribe 2 | ~~Pasa~~ por **la eco** ~~así~~, ~~el arranque~~ lo ~~hago~~ entre paciente y paciente.<br>_drops “que la radióloga te”_ |
| Soniox stt-rt-v5 | ~~Antes de terminar~~, ~~buenos días~~. Que la radióloga te lo hace entre paciente y paciente.<br>_invented phrase in place of “Pásate por la eco a las diez”_ |
| Deepgram Nova-3 | ~~De una hitosis~~, que la ~~la víctima del~~ paciente paciente.<br>_drops “eco a las diez”_<br>_drops “hace entre”_<br>_drops “y”_ |
| Deepgram Flux | _(no output)_ |
| Whisper large-v3-turbo | ~~Gracias~~ por la ~~hipótesis~~, que la ~~gran ayuda de la atención de pacientes~~ y ~~pacientes~~.<br>_drops “a las diez”_ |
| Voxtral Mini 4B Realtime | _(no output)_ |
| Cohere Transcribe | ~~Pasa de tu neitopsis~~, que la ~~radioma~~ te lo hace entre paciente y paciente.<br>_drops “a las diez”_ |

## Against three widely used open models

_Open source models_

Three widely used open-weights models ran on the same clips: OpenAI’s Whisper large-v3-turbo, Mistral’s Voxtral Mini 4B Realtime and Cohere Transcribe. On clinical audio, OmniScribe 2 gets 85.7% of medical terms right against 60–64% for the three. On real human speech, Cohere Transcribe is the strongest of the three: level with OmniScribe 2 on average, better on VoxPopuli parliament speech and worse on MediaSpeech. The [full leaderboard](results/leaderboard.md) has 22 open models; the best of them on clinical audio, Voxtral Small 24B, has the same clinical WER as OmniScribe 2 (8.5%) and gets 70.3% of medical terms right.

| System | Medical terms right | Clinical WER | FLEURS | VoxPopuli | MediaSpeech | Real-speech average |
|---|---:|---:|---:|---:|---:|---:|
| **OmniScribe 2** (Omniloy, self-hosted, not publicly available) | 85.7% | 8.5% | 2.9% | 8.3% | 8.4% | 6.6% |
| _Open-source models, default configuration_ | | | | | | |
| Whisper large-v3-turbo · open | 63.8% (ours better) | 10.3% (ours better) | 3.1% (level) | 13.5% (ours better) | 11.6% (ours better) | 9.4% |
| Voxtral Mini 4B Realtime · open | 59.9% (ours better) | 13.9% (ours better) | 3.1% (level) | 10.7% (ours better) | 10.5% (ours better) | 8.1% |
| Cohere Transcribe · open | 60.4% (ours better) | 9.3% (level) | 3.3% (level) | 6.6% (theirs better) | 10.5% (ours better) | 6.8% |

Open models run offline, like OmniScribe 2, in their default configuration: no prompt, and Spanish forced for Whisper and Cohere (Voxtral Realtime has no language setting). Whisper accepts a text prompt, which was not part of this comparison. Voxtral Realtime returned empty output on 7 clinical clips.

## Method and limits

_How to read this_

- **OmniScribe 2** is Omniloy's self-hosted transcription system. It runs on our own infrastructure and uses context from the patient's record. In this test that context included the medical terms spoken in each clip. That is a best case: a real record may not mention every term a clinician says.
- **Commercial APIs:** Soniox stt-rt-v5, Deepgram Nova-3 (Spanish) and Deepgram Flux (multilingual), called in their default configuration (Spanish set as the language, no custom vocabulary) and streamed in real time on their EU endpoints. Soniox and Deepgram also sell custom-vocabulary features; those were not part of this comparison.
- **Open-source models:** Whisper large-v3-turbo, Voxtral Mini 4B Realtime and Cohere Transcribe, run offline on a GPU with no prompt, and Spanish forced where the model allows it.
- **The clinical clips are synthetic voices** (ElevenLabs and Kokoro), degraded under control. We found no open recordings of real Spain-Spanish consultations, so real-clinic numbers will differ.
- **Modes differ.** OmniScribe 2 and the open models transcribed whole clips; the commercial APIs streamed in real time. We did not measure how much that matters.
- **The cases are hand-picked** to illustrate each pattern. The scoreboard and charts are computed on every clip, and the error marks come automatically from the same alignment the score uses. Significance: paired bootstrap with 2,000 resamples, resampling sentences on the clinical set and clips on real speech; the real-speech average is tested as a whole.
- **Full results** for every system are in the [GitHub repository](https://github.com/Omniloy/fonendo-bench) ([leaderboard](results/leaderboard.md)). **Audio:** the clips on this page are from the fonendo-bench clinical test set, © Omniloy, published for listening only (not for redistribution or training; see the dataset license). Voices: ElevenLabs and Kokoro-82M (Apache-2.0). Degraded clips mix in noise from MUSAN (Snyder, Chen and Povey, 2015; CC BY 4.0) and room impulse responses from OpenSLR SLR28 (Ko et al., 2017; Apache-2.0).

---

A public, reproducible benchmark of Spanish speech-to-text on clinical dictation and real Spanish speech, published by Omniloy. The systems the package runs are scored in their default configuration; OmniScribe 2 is a results-only row (see Method and limits).

- **Code, runners and scoring:** [github.com/Omniloy/fonendo-bench](https://github.com/Omniloy/fonendo-bench) (Apache-2.0).
- **Clinical audio:** gated dataset [huggingface.co/datasets/omniloy/fonendo-bench](https://huggingface.co/datasets/omniloy/fonendo-bench). Access is by request: email [info@omniloy.com](mailto:info@omniloy.com).
- **Public sets:** FLEURS, VoxPopuli and MediaSpeech are rebuilt from their original sources with `fonendo fetch`.
